import pandas as pd
import numpy as np

"""
BTC Delta Adjusted Market Making — Quantitative Strategy Module
===============================================================
Strategy: BTC Delta Adjusted Market Making
Ticker:   BTCUSDT
Type:     Mean-Reversion

Hypothesis:
    Exploits behavioral inefficiencies during extreme crypto sentiment phases.
    When the Fear & Greed proxy is at extremes (≤10 = extreme fear, ≥90 = extreme
    greed) AND realized volatility is above the 70th historical percentile, irrational
    retail activity creates asymmetric mean-reversion opportunities. The strategy
    takes contrarian directional positions (long on extreme fear, short on extreme
    greed) and exits once sentiment normalizes to the 25–75 band for two consecutive
    bars — mirroring the spread-reset rule of the underlying MM framework.

Signal Logic:
    fg_proxy        = percentile_rank(rolling_mean(log_ret, lookback)) × 100
    vol_percentile  = expanding_percentile_rank(rolling_std(log_ret, lookback))
    entry_long      = fg_proxy ≤ 10  AND vol_percentile > 0.70  → signal = +1
    entry_short     = fg_proxy ≥ 90  AND vol_percentile > 0.70  → signal = −1
    exit            = fg_proxy in (25, 75) for 2 consecutive bars → signal = 0

Data:
    Primary: Cybotrade — Fear & Greed Index, Realized Volatility (approximated
             from price; replace with live feed in production).
"""

import pandas as pd
import numpy as np

# ── Module-level constants ────────────────────────────────────────────────────
STRATEGY_NAME: str = "BTC Delta Adjusted Market Making"
TICKER: str = "BTCUSDT"
SIGNAL_TYPE: str = "mean_reversion"
LOOKBACK: int = 20  # Daily bars ≈ 100 hours (100h / 24h ≈ 4.2 days → 20 for stability)

# ── Sentiment & volatility thresholds (per spec) ─────────────────────────────
_FG_FEAR_THRESHOLD: float = 10.0    # fg_proxy ≤ 10  → extreme fear
_FG_GREED_THRESHOLD: float = 90.0   # fg_proxy ≥ 90  → extreme greed
_FG_NORMAL_LO: float = 25.0         # normalization band lower bound
_FG_NORMAL_HI: float = 75.0         # normalization band upper bound
_VOL_PERCENTILE_THRESHOLD: float = 0.70   # must exceed 70th pct to trigger
_NORMALIZATION_BARS: int = 2        # consecutive bars required to confirm exit


def generate_signals(prices: pd.Series, lookback: int = LOOKBACK) -> pd.Series:
    """
    Generate mean-reversion signals for the BTC Delta Adjusted MM strategy.

    The Fear & Greed Index is synthesized from price data as a percentile rank
    of rolling mean log-returns (scaled 0–100). Realized volatility is the
    rolling standard deviation of log-returns, ranked against its own history.
    Positions are held until sentiment confirms two consecutive bars inside the
    25–75 normalization band.

    Parameters
    ----------
    prices : pd.Series
        Datetime-indexed series of BTCUSDT close prices.
    lookback : int, optional
        Rolling window in bars (default=20). Maps to ~100 hours at daily
        resolution (BTC trades ~24/7).

    Returns
    -------
    pd.Series
        Integer signal series aligned to ``prices.index``:
        ``+1``  — Long  (extreme fear: contrarian upward mean-reversion bet)
        ``−1``  — Short (extreme greed: contrarian downward mean-reversion bet)
        ``0``   — Flat  (sentiment within normal range, vol below threshold,
                         or insufficient history)

    Raises
    ------
    ValueError
        If ``prices`` is not a pd.Series or ``lookback`` is non-positive.

    Notes
    -----
    In production, replace ``fg_proxy`` with the live Cybotrade Fear & Greed
    feed to obtain the true 0–100 index and remove the price-based approximation.
    """
    # ── Input validation ──────────────────────────────────────────────────────
    if not isinstance(prices, pd.Series):
        raise ValueError("`prices` must be a pd.Series of close prices.")
    if lookback <= 0:
        raise ValueError("`lookback` must be a positive integer.")

    # Return all-flat if there is not enough history to compute any indicator
    min_required: int = lookback * 2
    if len(prices) < min_required:
        return pd.Series(0, index=prices.index, dtype=int, name="signal")

    # ── Step 1: Log returns ───────────────────────────────────────────────────
    # Use log returns for stationarity and additive compounding properties
    log_ret: pd.Series = np.log(prices / prices.shift(1))

    # ── Step 2: Fear & Greed proxy (0–100 scale) ──────────────────────────────
    # Rolling mean of log-returns captures short-term momentum direction;
    # expanding percentile-rank maps the entire observed history to [0, 100].
    # Low values (≤10) reflect persistent negative momentum → proxy for fear.
    # High values (≥90) reflect persistent positive momentum → proxy for greed.
    rolling_mean_ret: pd.Series = (
        log_ret.rolling(window=lookback, min_periods=lookback).mean()
    )
    fg_proxy: pd.Series = (
        rolling_mean_ret
        .expanding(min_periods=lookback)
        .rank(pct=True)
        * 100.0
    )

    # ── Step 3: Realized volatility percentile ────────────────────────────────
    # Rolling std ≈ realized volatility; expanding rank gives the historical
    # percentile so the threshold is regime-adaptive rather than hard-coded.
    realized_vol: pd.Series = (
        log_ret.rolling(window=lookback, min_periods=lookback).std()
    )
    vol_pct: pd.Series = (
        realized_vol
        .expanding(min_periods=lookback)
        .rank(pct=True)
    )

    # ── Step 4: Boolean regime flags ──────────────────────────────────────────
    extreme_fear: pd.Series = fg_proxy <= _FG_FEAR_THRESHOLD    # irrational panic
    extreme_greed: pd.Series = fg_proxy >= _FG_GREED_THRESHOLD  # irrational euphoria
    high_vol: pd.Series = vol_pct > _VOL_PERCENTILE_THRESHOLD   # vol above 70th pct

    # Normalization check: fg_proxy settled inside the 25–75 band
    in_normal_band: pd.Series = (fg_proxy > _FG_NORMAL_LO) & (fg_proxy < _FG_NORMAL_HI)

    # Confirmed normalization requires _NORMALIZATION_BARS consecutive normal bars
    # Shift by (N-1) bars and AND with current: True only when both current
    # and the prior bar are both in the normal band (2-bar rolling confirmation).
    confirmed_exit: pd.Series = in_normal_band.copy()
    for lag in range(1, _NORMALIZATION_BARS):
        confirmed_exit = confirmed_exit & in_normal_band.shift(lag, fill_value=False)

    # ── Step 5: Stateful signal generation ───────────────────────────────────
    # Use a forward-carry loop to properly implement the hold-until-exit rule:
    #   • Enter long/short when extreme regime is detected
    #   • Hold through transitional (non-extreme, non-normal) zones
    #   • Exit only on confirmed 2-bar normalization
    signals_arr: np.ndarray = np.zeros(len(prices), dtype=int)
    current_pos: int = 0  # tracks the live position between bars

    for i in range(len(prices)):
        # Skip bars without valid indicator values (warmup period)
        if pd.isna(fg_proxy.iloc[i]) or pd.isna(vol_pct.iloc[i]):
            signals_arr[i] = 0
            continue

        fear_active: bool = bool(extreme_fear.iloc[i])
        greed_active: bool = bool(extreme_greed.iloc[i])
        hv_active: bool = bool(high_vol.iloc[i])
        exit_confirmed: bool = bool(confirmed_exit.iloc[i])

        if fear_active and hv_active:
            # Extreme fear + elevated vol → go long (mean-reversion: expect recovery)
            current_pos = 1
        elif greed_active and hv_active:
            # Extreme greed + elevated vol → go short (mean-reversion: expect pullback)
            current_pos = -1
        elif exit_confirmed:
            # Sentiment confirmed normal for ≥2 consecutive bars → reset to flat
            # (mirrors the spread-reset rule: adjusted_spread reverts to base_spread)
            current_pos = 0
        # else: hold current_pos through transitional zone (10<fg<25 or 75<fg<90)

        signals_arr[i] = current_pos

    return pd.Series(signals_arr, index=prices.index, dtype=int, name="signal")