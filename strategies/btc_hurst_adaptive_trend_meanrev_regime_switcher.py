import pandas as pd
import numpy as np
from typing import List

"""
BTC Hurst-Adaptive Trend-MeanRev Regime Switcher
=================================================
Hypothesis:
    Bitcoin's hourly price series alternates between statistically persistent
    (Hurst H > 0.55) and anti-persistent (H < 0.45) regimes detectable in real
    time via a 72-bar rolling R/S Hurst estimator.  In persistent regimes,
    momentum signals carry positive expected value; in anti-persistent regimes,
    mean reversion dominates and momentum signals should be inverted.

Regime detection  : rolling 72-bar R/S Hurst exponent on hourly log-returns.
Momentum proxy    : RSI(14) — overbought (>60) ≡ mom_z > +1.0;
                              oversold  (<40) ≡ mom_z < -1.0.
Signal rules      :
    H > 0.55  → follow RSI direction  (trend mode)
    H < 0.45  → fade  RSI direction   (mean-reversion mode)
    0.45 ≤ H ≤ 0.55 → flat, no trade  (ambiguous regime)

Position sizing   : 2 % NAV, max 1 concurrent position, no leverage > 1×.
Fees              : 3.5 bps per side.
Exit              : signal reversal, 24-bar max hold, or −3 % stop-loss.

Literature        : Hurst (1951) R/S analysis; "Optimizing Returns Using the
                    Hurst Exponent and Q-Learning on Momentum and Mean Reversion
                    Strategies"; crypto adaptive regime papers (2022-2024).
"""

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
STRATEGY_NAME: str = "BTC Hurst-Adaptive Trend-MeanRev Regime Switcher"
TICKER: str = "BTCUSDT"
SIGNAL_TYPE: str = "rsi"
LOOKBACK: int = 14  # RSI period; mirrors the 14-bar default in the spec rationale

# Internal strategy parameters
_HURST_WINDOW: int = 72          # bars for rolling R/S Hurst estimation
_HURST_PERSIST: float = 0.55     # above → persistent (trending) regime
_HURST_ANTIPERSIST: float = 0.45 # below → anti-persistent (choppy) regime
_RS_LAGS: List[int] = list(range(4, 20))  # sub-lags 4–19 for R/S calculation

_RSI_OVERBOUGHT: float = 60.0    # RSI threshold ≡ mom_z > +1.0 σ
_RSI_OVERSOLD: float = 40.0      # RSI threshold ≡ mom_z < −1.0 σ

_MOM_NORM_WINDOW: int = 168      # 168-bar window for momentum z-score baseline
_STOP_LOSS_PCT: float = 0.03     # 3 % hard stop-loss (magnitude)
_MAX_HOLD_BARS: int = 24         # maximum holding period in bars
_FEE_BPS: float = 3.5            # per-side fee in basis points
_POSITION_SIZE_PCT: float = 0.02 # 2 % NAV per trade


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _rs_hurst_window(arr: np.ndarray) -> float:
    """
    Compute the R/S Hurst exponent for a single window of log-returns.

    Algorithm
    ---------
    1. For each sub-lag L in _RS_LAGS, take the most-recent L observations.
    2. Compute R/S = (max cumulative deviation − min) / std(sub).
    3. OLS slope of log(R/S) ~ log(L) gives the Hurst exponent H.

    Parameters
    ----------
    arr : np.ndarray
        1-D array of log-returns (length = _HURST_WINDOW).

    Returns
    -------
    float
        Hurst exponent estimate, or np.nan if computation is degenerate.
    """
    # Strip any NaN values that may appear at rolling-window edges
    arr = arr[~np.isnan(arr)]
    if len(arr) < max(_RS_LAGS):
        return np.nan

    log_lags: List[float] = []
    log_rs: List[float] = []

    for lag in _RS_LAGS:
        sub = arr[-lag:]  # use the most-recent `lag` returns

        std = np.std(sub, ddof=1)
        if std == 0.0:
            continue  # degenerate window — skip this lag

        # Mean-adjusted cumulative sum (profile)
        cum_dev = np.cumsum(sub - np.mean(sub))

        # Rescaled range: (max − min) / std
        r_over_s = (cum_dev.max() - cum_dev.min()) / std
        if r_over_s <= 0.0:
            continue

        log_lags.append(np.log(lag))
        log_rs.append(np.log(r_over_s))

    if len(log_lags) < 4:
        return np.nan  # too few valid lags for a reliable OLS fit

    # OLS slope on the log-log plot = Hurst exponent
    slope: float = float(np.polyfit(log_lags, log_rs, 1)[0])
    return slope


def _rolling_hurst(log_returns: pd.Series) -> pd.Series:
    """
    Apply _rs_hurst_window in a rolling fashion over _HURST_WINDOW bars.

    Parameters
    ----------
    log_returns : pd.Series
        Full series of hourly log-returns.

    Returns
    -------
    pd.Series
        Rolling Hurst exponent, NaN where insufficient history exists.
    """
    return log_returns.rolling(
        window=_HURST_WINDOW,
        min_periods=_HURST_WINDOW,
    ).apply(_rs_hurst_window, raw=True)


def _wilder_rsi(prices: pd.Series, period: int) -> pd.Series:
    """
    Compute RSI using Wilder's exponential smoothing (industry standard).

    Overbought / oversold thresholds at 60 / 40 are used downstream as
    proxies for the ±1.0 σ momentum z-score entry bands in the spec.

    Parameters
    ----------
    prices : pd.Series
        Close price series.
    period : int
        RSI lookback (alpha = 1 / period in Wilder's EMA).

    Returns
    -------
    pd.Series
        RSI values in [0, 100].
    """
    delta = prices.diff()

    gain = delta.clip(lower=0.0)   # positive moves
    loss = (-delta).clip(lower=0.0)  # positive loss magnitudes

    # Wilder smoothing: EMA with com = period − 1 (equiv. alpha = 1/period)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    # Avoid division by zero when avg_loss is exactly 0
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    return rsi.fillna(50.0)  # neutral fill where RS undefined


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_signals(prices: pd.Series, lookback: int = 14) -> pd.Series:
    """
    Generate integer trading signals for the BTC Hurst-Adaptive strategy.

    Regime detection (Hurst R/S) drives direction of the RSI momentum signal:
        - Persistent  (H > 0.55): RSI overbought → long;  oversold → short.
        - Anti-persist (H < 0.45): RSI overbought → short; oversold → long.
        - Neutral band (0.45 ≤ H ≤ 0.55): flat regardless of RSI.

    Only one position is active at a time; the signal series encodes intent
    (1 / −1 / 0); position sizing, stop-loss, and max-hold enforcement are
    handled by the execution layer using _STOP_LOSS_PCT and _MAX_HOLD_BARS.

    Parameters
    ----------
    prices : pd.Series
        Hourly close price series with a DatetimeIndex.  Must contain at
        least max(_HURST_WINDOW, _MOM_NORM_WINDOW) + lookback bars.
    lookback : int, optional
        RSI period (default 14).  Passed through from the strategy spec.

    Returns
    -------
    pd.Series
        Integer signal series aligned to ``prices.index``:
        +1 = long, −1 = short, 0 = flat.

    Notes
    -----
    The R/S Hurst rolling apply is O(N × |_RS_LAGS|).  For production use
    with continuous bar updates, consider Numba-accelerated _rs_hurst_window.
    """
    # Guard: insufficient history returns all-flat signals
    min_bars = max(_HURST_WINDOW, _MOM_NORM_WINDOW) + lookback + 1
    if len(prices) < min_bars:
        return pd.Series(0, index=prices.index, dtype=int)

    # ------------------------------------------------------------------
    # Step 1 — Log-return series  r_t = log(close_t / close_{t-1})
    # ------------------------------------------------------------------
    log_ret: pd.Series = np.log(prices / prices.shift(1))

    # ------------------------------------------------------------------
    # Step 2 — Rolling 72-bar R/S Hurst exponent
    #   H > 0.55 → persistent (trending)
    #   H < 0.45 → anti-persistent (mean-reverting)
    # ------------------------------------------------------------------
    hurst: pd.Series = _rolling_hurst(log_ret)

    # ------------------------------------------------------------------
    # Step 3 — RSI(lookback) as momentum signal proxy
    #   RSI > 60 ≡ mom_z > +1.0  →  raw momentum direction = +1
    #   RSI < 40 ≡ mom_z < −1.0  →  raw momentum direction = −1
    #   40 ≤ RSI ≤ 60            →  no momentum signal      =  0
    # ------------------------------------------------------------------
    rsi: pd.Series = _wilder_rsi(prices, period=lookback)

    raw_mom = pd.Series(0, index=prices.index, dtype=int)
    raw_mom[rsi > _RSI_OVERBOUGHT] = 1   # overbought → upward momentum
    raw_mom[rsi < _RSI_OVERSOLD] = -1    # oversold   → downward momentum

    # ------------------------------------------------------------------
    # Step 4 — Regime gate and direction assignment
    #   Persistent     (H > 0.55) : follow momentum  (+1 / −1)
    #   Anti-persistent (H < 0.45): fade  momentum   (−1 / +1)
    #   Neutral band               : flat             (0)
    # ------------------------------------------------------------------
    signals = pd.Series(0, index=prices.index, dtype=int)

    is_persistent = hurst > _HURST_PERSIST          # trending regime
    is_antipersist = hurst < _HURST_ANTIPERSIST      # choppy regime
    # Neutral band (0.45 ≤ H ≤ 0.55) remains 0 — no trade

    # Trend mode: signal aligns with RSI momentum direction
    signals[is_persistent] = raw_mom[is_persistent]

    # Mean-reversion mode: signal is the inverse of RSI momentum direction
    signals[is_antipersist] = -raw_mom[is_antipersist]

    # ------------------------------------------------------------------
    # Step 5 — Suppress signals where Hurst is NaN (warm-up period)
    # ------------------------------------------------------------------
    signals[hurst.isna()] = 0

    return signals.astype(int)