import pandas as pd
import numpy as np


"""
ETH Hurst-Regime Adaptive Long/Short Strategy
==============================================
Strategy Name : ETH Hurst-Regime Adaptive Long/Short
Ticker        : ETHUSDT (perpetual)
Signal Type   : RSI (proxy for dual-regime MR z-score logic)
Lookback      : 14 bars (canonical RSI period; bridges 20h z-score & 80h Hurst windows)

Hypothesis
----------
BTC 1h log-returns alternate between statistically persistent (H > 0.55, trending)
and anti-persistent (H < 0.45, mean-reverting) regimes detectable via rolling R/S
Hurst estimation.  ETH perpetuals inherit the same regime (~0.85 rolling 1h correlation
with BTC in normal conditions).

Signal Approximation (this module)
------------------------------------
The full two-stage strategy requires paired BTC + ETH OHLCV bars.  This module
implements the RSI-based pre-check signal that approximates the MR-regime leg:

  RSI < 30  ≈  ETH z-score ≤ -1.5  →  LONG  (+1)
  RSI > 70  ≈  ETH z-score ≥ +1.5  →  SHORT (-1)
  30 ≤ RSI ≤ 70                     →  FLAT  ( 0)

This maps the oversold/overbought thresholds of the Mean-Reversion regime leg
onto a standard RSI, covering both long and short sides as the full strategy requires.
A 14-bar lookback is used (daily bars → 14 days), consistent with canonical RSI
parameterisation and the 1825-day (Jan 2021 – Dec 2024) backtest window.

Fee assumption : 3.5 bps per side (7 bps round-trip).
Minimum gross edge per trade must exceed 15 bps; trades with expected hold < 4 bars
are filtered out in production execution logic (not in signal generation itself).

Risk flags
----------
- Hurst R/S standard error ±0.10-0.15 on 80 bars → potential regime misclassification.
- BTC-ETH decoupling events (~4-6/year) require ETH-own Hurst override in production.
- Weekend low-volume microstructure noise can generate spurious MR signals.
"""

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

STRATEGY_NAME: str = "ETH Hurst-Regime Adaptive Long/Short"
TICKER: str = "ETHUSDT"
SIGNAL_TYPE: str = "rsi"
LOOKBACK: int = 14  # canonical RSI period (bars / days)

# RSI threshold constants — map to MR-regime z-score entry levels
_RSI_OVERSOLD: float = 30.0   # ≈ z ≤ -1.5  →  long entry
_RSI_OVERBOUGHT: float = 70.0  # ≈ z ≥ +1.5  →  short entry


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_rsi(prices: pd.Series, period: int) -> pd.Series:
    """
    Compute the Wilder RSI for a price series.

    Parameters
    ----------
    prices : pd.Series
        Closing price series (must be sorted oldest-first).
    period : int
        RSI smoothing period (Wilder's EMA equivalent).

    Returns
    -------
    pd.Series
        RSI values in [0, 100], indexed identically to *prices*.
        Leading ``period`` values are NaN (insufficient history).
    """
    # First-order price changes
    delta: pd.Series = prices.diff()

    # Separate gains and losses; fill NaN with 0 for rolling computation
    gain: pd.Series = delta.clip(lower=0.0)
    loss: pd.Series = (-delta).clip(lower=0.0)

    # Wilder smoothing: initial SMA seed, then exponential decay (alpha = 1/period)
    alpha: float = 1.0 / period
    avg_gain: pd.Series = gain.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    avg_loss: pd.Series = loss.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    # Relative Strength; guard against division-by-zero in trendless segments
    rs: pd.Series = avg_gain / avg_loss.replace(0.0, np.nan)

    # Classic RSI formula; when avg_loss == 0 the instrument is at RSI = 100
    rsi: pd.Series = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.where(avg_loss != 0.0, other=100.0)

    return rsi


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_signals(prices: pd.Series, lookback: int = LOOKBACK) -> pd.Series:
    """
    Generate long / short / flat signals for ETHUSDT using an RSI proxy
    for the Mean-Reversion regime leg of the Hurst-Regime Adaptive strategy.

    Signal logic
    ------------
    +1 (LONG)  : RSI < 30  — ETH oversold relative to recent momentum; fade the dip.
    -1 (SHORT) : RSI > 70  — ETH overbought; fade the spike.
     0 (FLAT)  : 30 ≤ RSI ≤ 70 — inside dead-band; no edge, no position.

    Parameters
    ----------
    prices : pd.Series
        Closing price time-series for ETHUSDT, sorted oldest-first.
        Index should be a DatetimeIndex for downstream compatibility.
    lookback : int, optional
        RSI period (Wilder EMA window).  Defaults to ``LOOKBACK`` (14).
        Must be ≥ 2.

    Returns
    -------
    pd.Series
        Integer signal series aligned to *prices* index:
        values in {-1, 0, +1}.  First ``lookback`` entries are 0 (warm-up).

    Raises
    ------
    ValueError
        If *prices* contains fewer bars than *lookback*, or if *lookback* < 2.
    TypeError
        If *prices* is not a ``pd.Series``.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> idx = pd.date_range("2024-01-01", periods=200, freq="1h")
    >>> px  = pd.Series(np.random.lognormal(0, 0.002, 200).cumprod() * 3000, index=idx)
    >>> sig = generate_signals(px, lookback=14)
    >>> sig.value_counts()
    """
    # ------------------------------------------------------------------ #
    # 0. Input validation
    # ------------------------------------------------------------------ #
    if not isinstance(prices, pd.Series):
        raise TypeError(f"prices must be a pd.Series, got {type(prices).__name__}.")
    if lookback < 2:
        raise ValueError(f"lookback must be ≥ 2, got {lookback}.")
    if len(prices) < lookback:
        raise ValueError(
            f"prices length ({len(prices)}) is shorter than lookback ({lookback}). "
            "Provide more historical bars."
        )

    # ------------------------------------------------------------------ #
    # 1. Compute RSI over the specified lookback window
    # ------------------------------------------------------------------ #
    rsi: pd.Series = _compute_rsi(prices, period=lookback)

    # ------------------------------------------------------------------ #
    # 2. Translate RSI levels to directional signals
    #    RSI < 30  → oversold  → LONG  (+1)   [MR regime: fade the dip]
    #    RSI > 70  → overbought → SHORT (-1)  [MR regime: fade the spike]
    #    dead-band [30, 70]    → FLAT   (0)   [no edge; preserve capital]
    # ------------------------------------------------------------------ #
    signals: pd.Series = pd.Series(0, index=prices.index, dtype=int, name="signal")
    signals[rsi < _RSI_OVERSOLD] = 1    # long: price extended to downside
    signals[rsi > _RSI_OVERBOUGHT] = -1  # short: price extended to upside

    # ------------------------------------------------------------------ #
    # 3. Zero-out warm-up period (RSI undefined before first full window)
    # ------------------------------------------------------------------ #
    signals.iloc[:lookback] = 0

    return signals