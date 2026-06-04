import pandas as pd
import numpy as np

"""
ETH Hurst-Regime Adaptive Long/Short
=====================================

Strategy Summary
----------------
Detects whether the ETH/BTC 1h price series inhabits a statistically
persistent (trending, H > 0.55) or anti-persistent (mean-reverting,
H < 0.45) regime via rolling 80-bar R/S Hurst estimation, then activates
the appropriate signal mode:

  • MR Regime (H < 0.45)  — fade 20-bar price z-score extremes (|z| >= 1.5)
  • TR Regime (H > 0.55)  — follow 24h momentum with EMA-5/EMA-20 confirmation
  • Dead-band (0.45-0.55) — no position (uncertain regime)

Primary edge: single-mode retail systematic bots (fixed-parameter momentum OR
mean-reversion) experience maximum adverse selection at regime transitions.
This strategy switches polarity precisely at those inflection points — buying
what the liquidated momentum crowd is forced to sell, and selling into what
squeezed mean-reversion traders are covering.

Ticker:       ETHUSDT perpetual
Frequency:    1h bars
Signal type:  mean_reversion (dominant entry mechanism; z-score fade)
Fee budget:   7 bps round-trip; minimum viable edge 15 bps per trade
Risk:         -1.5% hard stop-loss per trade; max leverage 2x NAV; max 1 open trade
Gate 0:       CONDITIONAL PASS (confidence 3/5)
Stage 2 plan: Backtest Jan 2021 - Dec 2024; target Sharpe > 0.8 net-of-fees,
              max drawdown < 15% before capital allocation
"""

__all__ = [
    "STRATEGY_NAME",
    "TICKER",
    "SIGNAL_TYPE",
    "LOOKBACK",
    "generate_signals",
]

# ── Public strategy constants ─────────────────────────────────────────────────
STRATEGY_NAME: str = "ETH Hurst-Regime Adaptive Long/Short"
TICKER: str        = "ETHUSDT"
SIGNAL_TYPE: str   = "mean_reversion"
LOOKBACK: int      = 14   # contextual window in days (14 × 24 h = 336 bars)

# ── Private hyper-parameters (all calibrated at 1h bar frequency) ─────────────
_HURST_WINDOW:  int   = 80     # R/S rolling window; reference lag = n/2 = 40
_ZSCORE_WINDOW: int   = 20     # MR regime: z-score mean/std rolling window
_MOM_WINDOW:    int   = 24     # TR regime: momentum lookback (24 bars = 24h)
_EMA_FAST:      int   = 5      # TR regime: fast EMA span
_EMA_SLOW:      int   = 20     # TR regime: slow EMA span
_H_LOWER:       float = 0.45   # Hurst anti-persistent upper bound (MR below this)
_H_UPPER:       float = 0.55   # Hurst persistent lower bound    (TR above this)
_Z_ENTRY:       float = 1.5    # MR: minimum |z-score| magnitude to open a position
_MOM_ENTRY:     float = 0.008  # TR: minimum |momentum| to open a position (0.8%)
_MOM_SCALE:     float = 0.020  # TR: momentum normalisation divisor (2% = full scale)
_FACTOR_MIN:    float = 0.5    # universal entry gate: |factor| must exceed this
_STOP_LOSS:     float = 0.015  # hard per-trade stop-loss reference (1.5%)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_hurst_rs(returns: np.ndarray) -> float:
    """
    Estimate the Hurst exponent for a fixed-length return array via the
    classical R/S (rescaled-range) method at a single reference lag.

    Strategy specification formula:
        RS  = (max(cumsum(r - mu)) - min(cumsum(r - mu))) / sigma(r)
        H   = log(RS) / log(n / 2)      [reference lag = n/2; n=80 -> lag=40]

    Parameters
    ----------
    returns : np.ndarray
        1-D array of log-returns; must have length >= 8.

    Returns
    -------
    float
        Hurst exponent H in (0, 1):
          H ~= 0.5  -> GBM / random walk (no memory)
          H >  0.5  -> persistent / trending (positive autocorrelation)
          H <  0.5  -> anti-persistent / mean-reverting (negative autocorrelation)
        Returns np.nan on degenerate (zero-variance or flat) windows.
    """
    n = len(returns)
    if n < 8:
        return np.nan

    std = returns.std(ddof=1)
    if std < 1e-12:
        return np.nan                          # flat / zero-variance window

    # Cumulative sum of mean-centred returns (Brownian bridge numerator)
    cumdev   = np.cumsum(returns - returns.mean())
    rs_range = cumdev.max() - cumdev.min()     # range of the rescaled trajectory

    if rs_range <= 0.0:
        return np.nan

    # Single reference-lag Hurst: H = log(R/S) / log(n/2)
    ref_lag = n / 2.0                          # n=80 -> ref_lag=40 per specification
    return float(np.log(rs_range / std) / np.log(ref_lag))


def _rolling_hurst(log_returns: pd.Series, window: int) -> pd.Series:
    """
    Compute a rolling Hurst exponent series by applying ``_compute_hurst_rs``
    over successive ``window``-length windows.

    Complexity: O(n * window); suitable for batch backtests on 1h data.
    First ``window - 1`` values are NaN (warm-up period has insufficient history).

    Parameters
    ----------
    log_returns : pd.Series
        1h log-return series. NaN values are tolerated; windows that contain
        any NaN are skipped and produce NaN output.
    window : int
        Number of bars per Hurst estimation window (strategy default: 80).

    Returns
    -------
    pd.Series
        Rolling Hurst exponent indexed identically to ``log_returns``,
        named "hurst_rs".
    """
    arr    = log_returns.to_numpy(dtype=float)
    result = np.full(len(arr), np.nan, dtype=float)

    for i in range(window - 1, len(arr)):
        chunk = arr[i - window + 1 : i + 1]
        if np.isnan(chunk).any():
            continue                           # skip NaN-contaminated windows
        result[i] = _compute_hurst_rs(chunk)

    return pd.Series(result, index=log_returns.index, name="hurst_rs")


# ─────────────────────────────────────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────────────────────────────────────

def generate_signals(prices: pd.Series, lookback: int = 14) -> pd.Series:
    """
    Generate regime-adaptive long/short signals for ETHUSDT perpetuals.

    Two-stage pipeline
    ------------------
    Stage 1 — Hurst regime detection (rolling 80-bar R/S):
        H < 0.45  -> anti-persistent -> Mean-Reversion (MR) regime
        H > 0.55  -> persistent      -> Trend-Following (TR) regime
        0.45-0.55 -> uncertain dead-band -> flat (no position)

    Stage 2 — ETH signal conditioned on active regime:
        MR:  f_MR = clip(-z_20, -3, 3) / 3   (normalised fade; buy oversold)
             Entry gate: |z_score| >= 1.5
        TR:  f_TR = sign(mom_24) * clip(|mom_24| / 2%, 0, 1)
             Entry gate: |momentum| >= 0.8% AND EMA-5/EMA-20 direction agrees

    Regime selector:
        factor = f_MR  if H < 0.45
               = f_TR  if H > 0.55
               = 0     if 0.45 <= H <= 0.55  (dead-band -> flat)

    Universal entry gate: factor set to 0 if |factor| < 0.5
    Signal: sign(factor) in {-1, 0, +1}

    Note on ``lookback``
    --------------------
    ``lookback=14`` (days) is the interface convention for the Balaane Quant
    signal harness (14 days * 24h = 336 hourly bars of retained context).
    Internal strategy windows follow the specification verbatim and are NOT
    scaled by this parameter: 80h Hurst, 20h z-score, 24h momentum.

    Parameters
    ----------
    prices : pd.Series
        ETH (or BTC-proxy) 1h close prices; DatetimeIndex preferred.
        Minimum required length: max(80, 20, 24, 20) + 1 = 81 bars.
    lookback : int, optional
        Contextual lookback in trading days (default 14). Retained for
        harness interface consistency; does not override internal windows.

    Returns
    -------
    pd.Series
        Integer signal series aligned to ``prices.index``:
           1  -> long
          -1  -> short
           0  -> flat  (uncertain regime, sub-threshold factor, or warm-up)
    """
    prices   = prices.dropna()
    min_bars = max(_HURST_WINDOW, _ZSCORE_WINDOW, _MOM_WINDOW, _EMA_SLOW) + 1

    # Guard: return all-flat when there is insufficient historical data
    if len(prices) < min_bars:
        return pd.Series(0, index=prices.index, dtype=int)

    # ── 1. Log-returns: r[t] = log(close[t] / close[t-1]) ─────────────────
    log_ret = np.log(prices / prices.shift(1))   # NaN at first bar is intentional

    # ── 2. Stage 1 — rolling R/S Hurst exponent → regime classification ────
    hurst   = _rolling_hurst(log_ret, window=_HURST_WINDOW)
    # NaN Hurst (warm-up) produces False on both comparisons -> dead-band (flat)
    mr_mask = hurst < _H_LOWER    # anti-persistent regime: mean-reversion mode
    tr_mask = hurst > _H_UPPER    # persistent regime:      trend-following mode

    # ── 3. MR signal component — 20-bar price z-score (Stage 2a) ───────────
    roll_mean = prices.rolling(_ZSCORE_WINDOW).mean()
    roll_std  = prices.rolling(_ZSCORE_WINDOW).std(ddof=1).clip(lower=1e-12)
    z_score   = (prices - roll_mean) / roll_std

    # Normalised fade factor: long when price is far below mean (z << 0)
    f_mr = np.clip(-z_score, -3.0, 3.0) / 3.0              # -> [-1, +1]
    # Suppress signal below entry threshold (avoid trading near-mean noise)
    f_mr = f_mr.where(z_score.abs() >= _Z_ENTRY, other=0.0)

    # ── 4. TR signal component — 24h momentum + EMA confirmation (Stage 2b) ─
    momentum = prices / prices.shift(_MOM_WINDOW) - 1.0     # 24h return ratio

    ema_fast = prices.ewm(span=_EMA_FAST, adjust=False).mean()
    ema_slow = prices.ewm(span=_EMA_SLOW, adjust=False).mean()

    # Momentum factor: direction * magnitude scaled to [-1, +1]
    mom_mag = (momentum.abs() / _MOM_SCALE).clip(upper=1.0)
    f_tr    = np.sign(momentum) * mom_mag

    # EMA confirmation filter: require EMA direction to agree before entry
    long_confirmed  = (momentum >  _MOM_ENTRY) & (ema_fast > ema_slow)
    short_confirmed = (momentum < -_MOM_ENTRY) & (ema_fast < ema_slow)
    f_tr = f_tr.where(long_confirmed | short_confirmed, other=0.0)

    # ── 5. Regime-conditional factor: MR or TR or dead-band (0) ────────────
    factor = pd.Series(0.0, index=prices.index, dtype=float)
    # Pandas .where(cond, other): keeps value where cond=True, replaces where False
    factor = factor.where(~mr_mask, other=f_mr)   # inject f_mr at MR bars
    factor = factor.where(~tr_mask, other=f_tr)   # inject f_tr at TR bars
    # mr_mask and tr_mask are mutually exclusive (H cannot be < 0.45 AND > 0.55)

    # ── 6. Universal entry gate — minimum signal strength required ──────────
    factor = factor.where(factor.abs() >= _FACTOR_MIN, other=0.0)

    # ── 7. Discretise continuous factor to regime signal {-1, 0, +1} ────────
    signal = np.sign(factor).astype(int)
    return pd.Series(signal.values, index=prices.index, dtype=int)