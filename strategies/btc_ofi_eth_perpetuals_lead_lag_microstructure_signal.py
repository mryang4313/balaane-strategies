import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Strategy: BTC-OFI ETH Perpetuals Lead-Lag Microstructure Signal
# Module: btc_ofi_eth_leadlag.py
# ─────────────────────────────────────────────────────────────────────────────
"""
BTC-OFI ETH Perpetuals Lead-Lag Microstructure Signal
======================================================
Hypothesis:
    Bitcoin perpetual order flow imbalance (OFI), when sustained at extreme
    levels (>95th or <5th hourly percentile for 30 seconds), predicts ETH
    perpetual price direction within a 1-5 minute window. BTC is the dominant
    price-discovery venue in crypto; institutional flow hits BTC first due to
    superior depth and tighter spreads, and ETH market makers adjust quotes
    reactively with a 3-30 second lag as they re-hedge BTC delta.

Live Signal Logic (1-second data):
    OFI_t         = BidQty_t - AskQty_t              # signed top-of-book pressure
    OFI_30s_t     = Σ OFI_s over t-29..t             # 30s rolling sum
    pct_t         = rank(OFI_30s_t, window=3600)/3600 # empirical CDF (hourly)
    trigger_t     = 1 if pct_t > 0.95 or pct_t < 0.05
    signal_t      = sign(OFI_30s_t) × trigger_t
    pos_ETH_t     = signal_t × 0.005 × NAV            # TP +0.40%, SL -0.25%, max 5min

Daily-Bar Proxy (this module):
    - Lookback momentum (14-day ROC) stands in for 30s OFI directionality.
    - Rolling percentile rank (window = lookback × 3) mirrors the hourly CDF
      gate; threshold adapted from P95/P5 → P80/P20 to remain meaningful at
      daily resolution while preserving the extreme-filter spirit.
    - Signals: +1 (long ETH), -1 (short ETH), 0 (flat / cooldown proxy).

Execution Parameters (live):
    Size        : 0.5% NAV per trade, max 3× leverage
    Fees        : 3.5 bps per side (taker)
    Take-Profit : +0.40% from entry
    Stop-Loss   : -0.25% from entry
    Max Hold    : 5 minutes; force-close at market
    Cooldown    : 90 seconds after any exit
    Max Concurrent: 1 ETH position

Risk Flags:
    - INFRASTRUCTURE: 1-second order book data requires a 4-8 week build or
      Tardis.dev / Kaiko data purchase (~$600 for 90 days).
    - LATENCY: Requires sub-500ms execution; cloud execution may be inadequate
      without exchange co-location.
    - FLASH CRASH (Aug 2024): Documented inversion where BTC OFI spike caused
      ETH to drop MORE than BTC; -0.25% SL may not fully contain gap risk.
    - CROWDING: Signal is published in academic literature; Tier-1 HFT shops
      likely trade this already, compressing the exploitable window.

Data Sources:
    Live  : Binance FAPI WebSocket — BTCUSDT & ETHUSDT perpetual top-of-book
    Backup: OKX WebSocket; CCXT unified WebSocket
    Backtest: Tardis.dev or Kaiko 1s order book snapshots

Gate 0 Verdict : CONDITIONAL PASS (confidence 3/5)
Stage 1 Rec    : Purchase 90d Tardis.dev data → offline backtest → if
                 Sharpe > 1.5 net of fees, proceed to co-location evaluation.
                 Total time to production: 8-12 weeks.
"""

# ─── Module-level constants ───────────────────────────────────────────────────

STRATEGY_NAME: str = "BTC-OFI ETH Perpetuals Lead-Lag Microstructure Signal"
TICKER: str = "ETHUSDT"          # execution venue; BTC is signal source only
SIGNAL_TYPE: str = "momentum"    # directional follow (same direction as OFI)
LOOKBACK: int = 14               # daily-bar proxy for 1h intraday OFI window


# ─── Signal generation ────────────────────────────────────────────────────────

def generate_signals(prices: pd.Series, lookback: int = LOOKBACK) -> pd.Series:
    """
    Generate long/short/flat signals for the BTC-OFI ETH lead-lag strategy
    using daily ETHUSDT closing prices as a proxy for the live OFI signal.

    At daily resolution, the 14-day rate-of-change (ROC) serves as a
    directional proxy for the 30-second cumulative BTC OFI, and a rolling
    percentile rank (window = 3 × lookback) mirrors the hourly empirical CDF
    gate used in the live strategy.  Signals fire only when the ranked momentum
    exceeds the P80 / P20 thresholds — the daily-resolution analogue of the
    live strategy's P95 / P5 extreme filter.

    Parameters
    ----------
    prices : pd.Series
        Daily ETHUSDT closing prices indexed by date (timezone-naive or UTC).
        Minimum length required: lookback + 3 × lookback = 4 × lookback bars.
    lookback : int, optional
        Momentum look-back period in days (default 14).  Must be ≥ 1.

    Returns
    -------
    pd.Series
        Integer signal series aligned to *prices* index:
            +1  → enter / hold ETH perpetual LONG
            -1  → enter / hold ETH perpetual SHORT
             0  → flat (no position / cooldown)
        dtype: int64.  First (lookback + 3 × lookback - 1) bars are 0
        (insufficient history).

    Raises
    ------
    ValueError
        If *prices* is empty or *lookback* < 1.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> idx = pd.date_range("2024-01-01", periods=120, freq="D")
    >>> px  = pd.Series(np.random.lognormal(0, 0.02, 120).cumprod() * 2000, index=idx)
    >>> sigs = generate_signals(px, lookback=14)
    >>> sigs.value_counts()
    """

    # ── Input validation ──────────────────────────────────────────────────────
    if prices.empty:
        raise ValueError("prices must not be empty.")
    if lookback < 1:
        raise ValueError(f"lookback must be ≥ 1, got {lookback}.")

    # ── Step 1: Compute lookback-period rate of change (momentum proxy for OFI)
    # ROC_t = (price_t / price_{t-lookback}) - 1
    # In the live strategy this corresponds to the directional sign of OFI_30s_t:
    # positive momentum ↔ sustained buy pressure; negative ↔ sell pressure.
    roc: pd.Series = prices.pct_change(periods=lookback)

    # ── Step 2: Rolling percentile rank of momentum ───────────────────────────
    # rank_window = 3 × lookback mirrors the hourly (3600-bar) CDF window
    # used in the live strategy at 1-second granularity.
    # rank(pct=True) returns values in (0, 1] — equivalent to the live pct_t.
    rank_window: int = lookback * 3
    pct_rank: pd.Series = roc.rolling(window=rank_window).rank(pct=True)

    # ── Step 3: Apply extreme-filter gate (daily analogue of P95 / P5) ───────
    # Threshold adapted to P80 / P20: at daily resolution the distribution has
    # fatter tails and fewer observations, so a tighter cut preserves signal
    # selectivity while maintaining a comparable firing rate (~20% of bars vs
    # ~10% at 1-second frequency after 30s accumulation requirement).
    upper_threshold: float = 0.80   # → long  (live: 0.95)
    lower_threshold: float = 0.20   # → short (live: 0.05)

    # ── Step 4: Assign directional signals ───────────────────────────────────
    # signal_t = sign(momentum) × trigger_t
    # Flat (0) when:  (a) percentile rank is in neutral zone [0.20, 0.80], OR
    #                 (b) insufficient history (NaN rank) — warm-up period.
    signals: pd.Series = pd.Series(0, index=prices.index, dtype=int)

    long_condition: pd.Series = pct_rank > upper_threshold   # extreme buy pressure
    short_condition: pd.Series = pct_rank < lower_threshold  # extreme sell pressure

    signals.loc[long_condition] = 1    # enter ETH LONG
    signals.loc[short_condition] = -1  # enter ETH SHORT

    # Mask warm-up bars where the rolling rank has not yet converged to NaN-free
    signals.loc[pct_rank.isna()] = 0

    # ── Step 5: Suppress signal flips within the cooldown proxy window ────────
    # Live strategy enforces 90-second post-exit cooldown to prevent signal
    # chasing.  At daily resolution we replicate this by requiring a signal to
    # persist for at least 1 bar before allowing a direction reversal — a
    # direction change that reverses on the very next bar is zeroed out.
    prev_signals: pd.Series = signals.shift(1).fillna(0).astype(int)
    instant_reversal: pd.Series = (signals != 0) & (prev_signals != 0) & (signals != prev_signals)
    signals.loc[instant_reversal] = 0  # zero the flip bar; re-enter on next bar

    return signals.astype(int)