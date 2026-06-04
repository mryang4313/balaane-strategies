import pandas as pd
import numpy as np
from typing import Optional

"""
ETH Hurst-Regime Adaptive Long/Short
======================================
Hypothesis:
    BTC's 1h return series alternates between statistically persistent
    (H > 0.55, trending) and anti-persistent (H < 0.45, mean-reverting)
    regimes, detectable in real time via rolling R/S Hurst estimation over
    an 80-bar window.  ETH perpetuals (~0.85 rolling 1h return correlation
    with BTC) inherit the same regime.  Conditioning signal polarity on the
    real-time Hurst regime extracts premium from BOTH regime types:
        • Anti-persistent (H < 0.45) → mean-reversion: fade ETH z-score extremes
        • Persistent      (H > 0.55) → trend-following: ride ETH momentum
        • Dead-band       (0.45–0.55) → flat: uncertain regime, no edge taken

Signal proxy (this module):
    SIGNAL_TYPE = "rsi" with lookback=14 acts as a single-series structural
    proxy for the two-stage BTC-regime / ETH-signal logic:
        RSI < 30  →  +1 long   (MR mode: oversold, analogous to z ≤ −1.5)
        RSI > 70  →  −1 short  (MR mode: overbought, analogous to z ≥ +1.5)
        RSI 30–70 →   0 flat   (mirrors Hurst dead-band, no confirmed edge)
    An in-series Hurst filter runs on ETH log-returns as a regime proxy;
    production use should supply BTC close separately for true regime detection.

Position sizing (production):
    Notional = |factor| × 3% NAV, capped at 3% NAV.  Max 1 concurrent trade.
    Max leverage 2× NAV.  Hard stop-loss −1.5% per trade.

Fee assumption:
    3.5 bps/side (7 bps round-trip).  Minimum gross edge per trade: 15 bps.
    Skip trades with expected hold < 4 bars.

Regime exit override:
    If Hurst crosses the opposite threshold mid-trade, flatten immediately
    regardless of P&L.

Risk flags:
    • Hurst R/S SE ±0.10–0.15 on 80 bars → ~18–22% false regime labels.
    • Fee drag in active TR regimes can reach 42–70 bps/day.
    • BTC–ETH decoupling during ETH-specific events (4–6×/year).
    • Weekend microstructure noise can generate spurious H < 0.45 readings.

Gate 0 verdict: CONDITIONAL PASS (confidence 3/5).
Stage 1 action: backtest on 1h BTC/ETH parquet Jan 2021–Dec 2024.
"""

# ── Module-level constants ────────────────────────────────────────────────────

STRATEGY_NAME: str = "ETH Hurst-Regime Adaptive Long/Short"
TICKER: str = "ETHUSDT"
SIGNAL_TYPE: str = "rsi"
LOOKBACK: int = 14  # RSI lookback; approximates 20–24h signal windows on daily bars

# RSI thresholds that mirror MR-regime entry levels (z ≤ −1.5 / z ≥ +1.5)
_RSI_OVERSOLD: float = 30.0    # long entry threshold  (MR mode)
_RSI_OVERBOUGHT: float = 70.0  # short entry threshold (MR mode)

# Hurst dead-band — uncertain regime: no position taken
_HURST_MR_THRESHOLD: float = 0.45   # below → anti-persistent, mean-revert
_HURST_TR_THRESHOLD: float = 0.55   # above → persistent, trend-follow
_HURST_WINDOW: int = 80             # R/S estimation window (bars)
_HURST_REF_LAG: float = 40.0        # reference lag = window / 2


# ── Helper: Wilder RSI ────────────────────────────────────────────────────────

def _compute_rsi(prices: pd.Series, period: int) -> pd.Series:
    """
    Compute Wilder's Relative Strength Index.

    Parameters
    ----------
    prices : pd.Series
        Close price series indexed by timestamp.
    period : int
        Lookback window for smoothing (standard: 14).

    Returns
    -------
    pd.Series
        RSI values in [0, 100]; NaN for the first ``period`` warm-up bars.
    """
    delta: pd.Series = prices.diff()  # bar-to-bar price change Δ[t]

    # Split changes into gains (+) and losses (+); losses stored as positives
    gains: pd.Series = delta.clip(lower=0.0)
    losses: pd.Series = (-delta).clip(lower=0.0)

    # Wilder smoothing: exponential MA with α = 1/period
    avg_gain: pd.Series = gains.ewm(
        alpha=1.0 / period, min_periods=period, adjust=False
    ).mean()
    avg_loss: pd.Series = losses.ewm(
        alpha=1.0 / period, min_periods=period, adjust=False
    ).mean()

    # RS = avg_gain / avg_loss; guard against zero-loss (avoid division by zero)
    rs: pd.Series = avg_gain / avg_loss.replace(0.0, np.nan)

    # Classic RSI formula: 100 - 100/(1+RS)
    rsi: pd.Series = 100.0 - (100.0 / (1.0 + rs))
    return rsi.rename("rsi")


# ── Helper: rolling R/S Hurst exponent ───────────────────────────────────────

def _compute_hurst_rs(
    log_returns: pd.Series,
    window: int = _HURST_WINDOW,
    ref_lag: float = _HURST_REF_LAG,
) -> pd.Series:
    """
    Rolling R/S (rescaled-range) Hurst exponent estimator.

    For each trailing window of log-returns:
        R/S = (max(CumDev) − min(CumDev)) / std(returns)
        H   = log(R/S) / log(ref_lag)        # ref_lag = window / 2

    Interpretation:
        H < 0.45  → anti-persistent (mean-reverting)
        H > 0.55  → persistent (trending)
        0.45–0.55 → uncertain (dead-band)

    Note: Standard error ±0.10–0.15 on 80 bars; validate thresholds in
    backtest via sensitivity grid (60h / 80h / 120h, 0.40/0.60 alternate
    thresholds) per Stage 1 recommendation.

    Parameters
    ----------
    log_returns : pd.Series
        Bar-frequency log-returns (ln(close[t] / close[t-1])).
    window : int
        Rolling estimation window in bars (default 80).
    ref_lag : float
        Reference lag for normalisation (default window / 2 = 40).

    Returns
    -------
    pd.Series
        Hurst exponent per bar; NaN for warm-up bars or degenerate windows.
    """

    def _rs_single(sub: np.ndarray) -> float:
        """Compute one H estimate for a 1-D array of log-returns."""
        if len(sub) < 8:
            return np.nan  # insufficient data for reliable estimate

        std_s: float = float(np.std(sub, ddof=1))
        if std_s == 0.0:
            return np.nan  # flat price — rescaling undefined

        mean_r: float = float(np.mean(sub))
        cum_dev: np.ndarray = np.cumsum(sub - mean_r)   # cumulative deviation from mean

        r_range: float = float(np.max(cum_dev) - np.min(cum_dev))  # range R
        if r_range <= 0.0:
            return np.nan

        rs: float = r_range / std_s                          # rescaled range RS
        return float(np.log(rs) / np.log(ref_lag))           # H = log(RS) / log(n/2)

    hurst_vals: pd.Series = (
        log_returns
        .rolling(window=window, min_periods=window)
        .apply(_rs_single, raw=True)
    )
    return hurst_vals.rename("hurst")


# ── Primary signal function ───────────────────────────────────────────────────

def generate_signals(prices: pd.Series, lookback: int = LOOKBACK) -> pd.Series:
    """
    Generate regime-adaptive long/short signals for ETHUSDT.

    Two-stage logic (single-series proxy mode)
    -------------------------------------------
    Stage 1 — Regime detection via rolling R/S Hurst on ETH log-returns
    (production: supply BTC log-returns; ETH is used here as a structural
    proxy when only one series is available):
        H < 0.45  → MR  regime: RSI signals used as-is   (fade extremes)
        H > 0.55  → TR  regime: RSI signal polarity flip  (follow breakouts)
        0.45–0.55 → Dead-band: flat regardless of RSI

    Stage 2 — ETH signal via RSI(lookback):
        RSI < 30  →  +1 raw signal  (oversold: MR long  / TR short via flip)
        RSI > 70  →  −1 raw signal  (overbought: MR short / TR long via flip)
        30–70     →   0 raw signal  (neutral)

    Parameters
    ----------
    prices : pd.Series
        ETHUSDT close prices indexed by timestamp (1h or daily bars).
        Must contain at least ``lookback + 1`` non-NaN observations.
    lookback : int, optional
        RSI window length in bars (default 14).

    Returns
    -------
    pd.Series
        Integer signal series aligned to ``prices`` index:
            +1  long entry signal
            -1  short entry signal
             0  flat (no position)
        dtype: int64.  Warm-up bars and uncertain-regime bars return 0.

    Raises
    ------
    ValueError
        If ``prices`` is empty or ``lookback < 2``.
    """
    # ── Input validation ──────────────────────────────────────────────────────
    if prices.empty:
        raise ValueError("prices series must not be empty.")
    if lookback < 2:
        raise ValueError(f"lookback must be ≥ 2; received {lookback}.")

    # ── Step 1: RSI on ETH close — core mean-reversion signal generator ───────
    # Wilder RSI(14); first `lookback` bars are NaN (warm-up unavoidable)
    rsi: pd.Series = _compute_rsi(prices, period=lookback)

    # ── Step 2: Base RSI signal — MR-mode polarity ────────────────────────────
    # RSI < 30 → +1 (oversold extreme, analogous to z ≤ −1.5)
    # RSI > 70 → −1 (overbought extreme, analogous to z ≥ +1.5)
    # [30, 70] →  0 (neutral, mirrors Hurst dead-band)
    signals: pd.Series = pd.Series(0, index=prices.index, dtype=int, name="signal")
    signals[rsi < _RSI_OVERSOLD] = 1     # oversold → long (MR default)
    signals[rsi > _RSI_OVERBOUGHT] = -1  # overbought → short (MR default)

    # ── Step 3: Hurst regime filter on ETH log-returns (proxy for BTC) ────────
    # Production note: replace log_ret with BTC log-returns for true regime
    # detection per strategy spec (BTC is regime detector, ETH is traded asset)
    log_ret: pd.Series = np.log(prices / prices.shift(1))  # ln(close[t]/close[t-1])

    # Use min of 80-bar spec and half the series to avoid degenerate windows
    hurst_window: int = max(20, min(_HURST_WINDOW, len(prices) // 2))
    hurst: pd.Series = _compute_hurst_rs(
        log_ret, window=hurst_window, ref_lag=hurst_window / 2.0
    )

    # ── Step 4: TR regime — flip RSI polarity to trend-following ──────────────
    # In persistent markets: oversold breakdowns → short; overbought breakouts → long
    # Flip: original +1 (oversold MR long) becomes −1 (oversold TR short), etc.
    trend_mask: pd.Series = hurst > _HURST_TR_THRESHOLD
    signals[trend_mask] = -signals[trend_mask]  # polarity reversal in TR mode

    # ── Step 5: Dead-band and warm-up enforcement — force flat ────────────────
    # Uncertain regime (0.45 ≤ H ≤ 0.55): no reliable regime → no position
    dead_band_mask: pd.Series = (hurst >= _HURST_MR_THRESHOLD) & (hurst <= _HURST_TR_THRESHOLD)
    signals[dead_band_mask] = 0

    # Hurst NaN (insufficient history for 80-bar window) → flat
    signals[hurst.isna()] = 0

    # RSI NaN (insufficient history for lookback bars) → flat
    signals[rsi.isna()] = 0

    # ── Step 6: Final dtype and name enforcement ──────────────────────────────
    signals = signals.astype(int)
    signals.name = "signal"

    return signals