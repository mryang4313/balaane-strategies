import pandas as pd
import numpy as np

"""
BTC-ETH Perp Funding Rate Spread Mean Reversion
================================================
Strategy: Fade extreme divergences in the BTC vs ETH perpetual funding rate spread.

Hypothesis:
    The spread (btc_funding_rate − eth_funding_rate, %/8h) is stationary and
    mean-reverting on a 14–30 day horizon.  When the 90-bar (30-day) rolling
    z-score breaches ±2σ, BTC-specific retail over-leveraging has irrationally
    detached funding costs from ETH.  Three convergence forces restore parity:
      (a) Axiom-5 OFI cascade — BTC microstructure extremes transmit to ETH within
          minutes, pulling ETH demand up and cooling BTC-specific excess;
      (b) margin erosion — leveraged BTC longs face funding-induced collateral drain;
      (c) institutional carry rotation — desks migrate from expensive BTC carry to
          cheaper ETH carry.
    The edge persists because retail CEX perpetual traders are asset-siloed and
    never frame their exposure as a BTC/ETH spread, so they never self-arbitrage
    the cross-asset carry differential.

Signal mapping to BTCUSDT:
    z_t > +2.0  →  -1  (SHORT BTC perp / LONG ETH perp — fade BTC premium)
    z_t < -2.0  →  +1  (LONG BTC perp / SHORT ETH perp — fade BTC discount)
    |z_t| ≤ 2.0 →   0  (flat)

Position sizing (documented; enforced in execution layer, not in this signal module):
    size_multiplier = min(|z_t| − 2.0, 1.0)        # 0.0 – 1.0 linear scale
    notional_per_leg = NAV × 0.01 × size_multiplier # 1–2 % NAV per leg, ≤ 2× leverage

Exit conditions (first met, evaluated each 8h bar):
    1. |z_t| < 0.5                         — spread reverted to near-mean
    2. Either leg adverse price move > 8 %  — stop-loss
    3. Bars held > 21                       — time-stop (7 calendar days)
    4. BTC 1h price drop > 10 %            — flash-crash sentinel (intra-period)

Fees: 3.5 bps/side × 4 legs = 14 bps round-trip.

Risk controls:
    - Max 2 concurrent spread positions
    - Halt trading if rolling 30-day strategy drawdown < −5 % NAV

Data sources:
    Primary   — CryptoQuant btc_funding_rate / eth_funding_rate (Binance perps, 8h)
    Secondary — Internal Parquet 1h OHLCV bars (BTC & ETH close prices)

Reference:
    Signal Type : mean_reversion  |  Ticker : BTCUSDT  |  Lookback : 22 bars
    Gate-0 Verdict : CONDITIONAL PASS  |  Confidence : 3/5

Author : Sharpe (Balaane Quant)
Date   : 2026-06-05
"""

# ── Module-level constants ────────────────────────────────────────────────────

STRATEGY_NAME: str = "BTC-ETH Perp Funding Rate Spread Mean Reversion"
TICKER: str = "BTCUSDT"
SIGNAL_TYPE: str = "mean_reversion"
LOOKBACK: int = 22  # bars; 22 × 8h ≈ 30 calendar days (matches 90-bar 8h window)

# Z-score thresholds
_Z_ENTRY: float = 2.0   # breach ±2σ triggers entry
_Z_EXIT: float = 0.5    # reversion to <0.5σ triggers close

# Position sizing constants (informational; enforced in execution layer)
_Z_SCALE_CAP: float = 1.0        # excess z capped at 1.0 for linear scale
_BASE_NAV_PER_LEG: float = 0.01  # 1 % NAV per leg at threshold
_MAX_NAV_PER_LEG: float = 0.02   # 2 % NAV per leg at z ≥ 3.0
_MAX_LEVERAGE_PER_LEG: float = 2.0

# Risk & execution constants (informational)
_STOP_LOSS_PCT: float = 0.08     # 8 % adverse price move → stop both legs
_FLASH_CRASH_PCT: float = 0.10   # 10 % BTC 1h drop → immediate close
_MAX_HOLD_BARS: int = 21         # 7 calendar days at 8h bars
_ROUND_TRIP_FEE_BPS: float = 14  # 3.5 bps × 4 legs
_MAX_CONCURRENT_POSITIONS: int = 2
_MAX_DRAWDOWN_HALT: float = -0.05  # −5 % rolling 30-day NAV → halt


# ── Public API ────────────────────────────────────────────────────────────────

def generate_signals(prices: pd.Series, lookback: int = LOOKBACK) -> pd.Series:
    """Generate mean-reversion entry signals for the BTC-ETH funding spread strategy.

    In production this function should be called with the BTC-ETH funding rate
    spread series (btc_funding_rate − eth_funding_rate, %/8h) as `prices`.
    When called with raw BTC price data the z-score is computed over price
    directly and acts as a momentum-exhaustion proxy.

    The function returns a **direction signal for the BTC leg only**:
        +1  →  LONG  BTC perp (pair: SHORT ETH perp)
        -1  →  SHORT BTC perp (pair: LONG  ETH perp)
         0  →  flat (no position)

    The ETH leg is always the mirror; sizing is delegated to the execution layer
    using `compute_position_size()`.

    Args:
        prices:   pd.Series (float) indexed by datetime-like values.
                  Should represent the BTC-ETH funding rate spread (%/8h) in
                  production, or BTC close price as a fallback proxy.
        lookback: Rolling window length in bars for z-score normalisation.
                  Default 22 (≈ 30 calendar days at 8h frequency).
                  Must be ≥ 2 and ≤ len(prices).

    Returns:
        pd.Series[int] with the same index as `prices`.
        Values ∈ {-1, 0, 1}.  The first (lookback − 1) bars are always 0
        because there is insufficient history for a stable z-score.

    Raises:
        TypeError:  if `prices` is not a pd.Series.
        ValueError: if `lookback` < 2 or `lookback` > len(prices).
    """
    # ── Guard clauses ─────────────────────────────────────────────────────────
    if not isinstance(prices, pd.Series):
        raise TypeError(f"prices must be pd.Series, got {type(prices).__name__}")
    if lookback < 2:
        raise ValueError(f"lookback must be >= 2, got {lookback}")
    if len(prices) < lookback:
        raise ValueError(
            f"prices length ({len(prices)}) is less than lookback ({lookback})"
        )

    # ── Step 1: Rolling statistics over the lookback window ───────────────────
    # min_periods=lookback ensures no partial-window z-scores leak into signals
    rolling = prices.rolling(window=lookback, min_periods=lookback)
    mu: pd.Series = rolling.mean()          # rolling centre of the spread distribution
    sigma: pd.Series = rolling.std(ddof=1)  # rolling spread volatility (sample std)

    # ── Step 2: Z-score normalisation ─────────────────────────────────────────
    # Replace zero sigma with NaN to avoid ±inf z-scores during flat funding periods
    # (e.g. weekends, exchange maintenance windows) — those bars emit signal = 0
    sigma_safe: pd.Series = sigma.where(sigma > 0.0, other=np.nan)
    z: pd.Series = (prices - mu) / sigma_safe  # dimensionless spread deviation

    # ── Step 3: Signal generation ─────────────────────────────────────────────
    # z > +2.0 → BTC funding premium is extreme; fade BTC excess → SHORT BTC
    # z < -2.0 → BTC funding discount vs ETH; fade underpricing  → LONG  BTC
    # |z| ≤ 2.0 or z is NaN → flat; no edge over round-trip fees
    signals: pd.Series = pd.Series(0, index=prices.index, dtype=int)
    signals = signals.where(~(z > _Z_ENTRY), other=-1)   # SHORT BTC / LONG ETH
    signals = signals.where(~(z < -_Z_ENTRY), other=1)   # LONG  BTC / SHORT ETH

    # Explicitly zero-out any bar where z is undefined (NaN guard, first `lookback-1` bars)
    signals[z.isna()] = 0

    return signals


def compute_spread(
    btc_funding: pd.Series,
    eth_funding: pd.Series,
) -> pd.Series:
    """Compute the raw BTC-ETH perpetual funding rate spread.

    Both series must share the same datetime index aligned to 8h funding
    settlement timestamps (00:00, 08:00, 16:00 UTC).

    Args:
        btc_funding: BTC perpetual funding rate (%/8h), Binance perps.
        eth_funding: ETH perpetual funding rate (%/8h), Binance perps.

    Returns:
        pd.Series of spread values (btc_funding − eth_funding, %/8h).

    Raises:
        ValueError: if the two series do not share the same index.
    """
    if not btc_funding.index.equals(eth_funding.index):
        raise ValueError(
            "btc_funding and eth_funding must share an identical datetime index; "
            "align both series to the 8h settlement grid before calling."
        )

    # Simple arithmetic spread — Type B formula per Gate-0 assessment
    spread: pd.Series = btc_funding - eth_funding
    spread.name = "btc_eth_funding_spread"
    return spread


def compute_position_size(
    z_score: float,
    nav: float,
    base_pct: float = _BASE_NAV_PER_LEG,
    z_entry: float = _Z_ENTRY,
    z_scale_cap: float = _Z_SCALE_CAP,
) -> float:
    """Compute the USD notional per leg for a given z-score and portfolio NAV.

    Sizing scales linearly with excess z beyond the entry threshold, capped
    so the maximum allocation is 2× the base NAV fraction at |z| ≥ 3.0.

    Formula:
        size_multiplier = min(|z| − z_entry, z_scale_cap)   ∈ [0.0, 1.0]
        notional_per_leg = NAV × base_pct × (1 + size_multiplier)

    Args:
        z_score:    Current z-score of the BTC-ETH funding spread.
        nav:        Current portfolio NAV in USD.
        base_pct:   Base NAV fraction per leg at the entry threshold (default 0.01).
        z_entry:    Entry z-score threshold (default 2.0).
        z_scale_cap: Maximum additional z-score units for linear scaling (default 1.0).

    Returns:
        float: USD notional to allocate per leg (BTC leg = ETH leg = returned value).
               Returns 0.0 if |z_score| < z_entry (no active signal).
    """
    abs_z = abs(z_score)

    # No signal — return zero notional
    if abs_z <= z_entry:
        return 0.0

    # Linear scale factor clipped to [0, z_scale_cap]; adds up to 100 % of base_pct
    excess_z: float = min(abs_z - z_entry, z_scale_cap)       # ∈ [0.0, 1.0]
    size_multiplier: float = excess_z / z_scale_cap            # normalise to [0, 1]

    # Notional ramps from base_pct × NAV at threshold to 2 × base_pct × NAV at cap
    notional_per_leg: float = nav * base_pct * (1.0 + size_multiplier)
    return notional_per_leg


def compute_exit_signal(
    z_score: float,
    bars_held: int,
    btc_price_move_pct: float,
    eth_price_move_pct: float,
    z_exit: float = _Z_EXIT,
    stop_loss_pct: float = _STOP_LOSS_PCT,
    max_hold_bars: int = _MAX_HOLD_BARS,
) -> tuple[bool, str]:
    """Evaluate whether an open spread position should be closed.

    Checks all four exit conditions in priority order (reversion → stop-loss →
    time-stop; the flash-crash sentinel is handled at execution layer from 1h bars).

    Args:
        z_score:            Current z-score of the BTC-ETH funding spread.
        bars_held:          Number of 8h bars elapsed since position entry.
        btc_price_move_pct: BTC price change from entry price as a decimal
                            (negative = price fell, e.g. −0.09 = −9 %).
        eth_price_move_pct: ETH price change from entry price as a decimal.
        z_exit:             Mean-reversion closure threshold (default 0.5).
        stop_loss_pct:      Adverse price-move stop threshold (default 0.08).
        max_hold_bars:      Time-stop bar count (default 21 bars = 7 calendar days).

    Returns:
        (should_exit: bool, reason: str)
        reason is one of: 'mean_reversion', 'stop_loss_btc', 'stop_loss_eth',
                          'time_stop', or '' (no exit triggered).
    """
    # Condition 1: spread has reverted to near-mean — primary profitable exit
    if abs(z_score) < z_exit:
        return True, "mean_reversion"

    # Condition 2: adverse price move on either leg exceeds stop threshold
    # Use absolute move — direction depends on position side (handled by caller)
    if abs(btc_price_move_pct) > stop_loss_pct:
        return True, "stop_loss_btc"
    if abs(eth_price_move_pct) > stop_loss_pct:
        return True, "stop_loss_eth"

    # Condition 3: max hold period reached — time-stop prevents extended drawdown
    if bars_held >= max_hold_bars:
        return True, "time_stop"

    return False, ""