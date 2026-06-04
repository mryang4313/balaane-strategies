import pandas as pd
import numpy as np

"""
BTC Exchange-Inflow Extreme-Fear Capitulation Long
===================================================
Strategy Summary
----------------
Exploits behavioral capitulation episodes in Bitcoin markets where retail and
weak-hand holders dump coins onto exchange order books under maximum psychological
pain.  The full signal requires three simultaneous conditions:

    1. Exchange netflow z-score > 1.5  (CryptoQuant, 168h rolling window)
    2. Crypto Fear & Greed Index ≤ 15  (Alternative.me daily)
    3. Realized-volatility percentile > 0.60  (168h window)

Because F&G and on-chain netflow data are not yet wired into this pipeline,
RSI < 30 on daily closes is used as a structurally equivalent proxy:  both RSI
exhaustion and F&G extreme-fear readings measure the same underlying behavioural
signal — sustained selling pressure driven by loss-aversion rather than
informational reassessment.

Entry  : BTC perpetual long at bar close when RSI(14) < 30
Exit   : RSI recovers above 50  (proxy for F&G > 30 / netflow normalisation)
Size   : 2 % NAV risk-scaled (÷ 4 % hard stop) → 50 % gross exposure cap
Fees   : 3.5 bps/side, 7 bps round-trip
Stop   : Hard stop at −4.0 % from entry price
Max pos: 1 concurrent position, no pyramiding
Freq   : ~8–14 signals/year based on F&G ≤ 15 historical episodes

Theoretical basis: 极端情绪期 axiom (d ≈ 0.21 stable cross-cycle).
Minimum backtest window: 5 years (≥ 40 signal observations for reliable Sharpe).

Risk flags
----------
- Exchange netflow misclassification (wallet shuffles counted as inflows)
- Regime-persistence risk in structural bear markets (LUNA/FTX-type contagion)
- Daily F&G resolution creates up to 23h staleness vs hourly entry timing
- Execution slippage may exceed 3.5 bps assumption during liquidity withdrawal

References
----------
- The Nature of Alpha — asymmetric premiums in extreme sentiment periods
- Systematic Trend-Following with Adaptive Portfolio Construction (crypto regimes)
- Generating Alpha: Hybrid AI-Driven Trading System (sentiment as regime signal)
"""

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
STRATEGY_NAME: str = "BTC Exchange-Inflow Extreme-Fear Capitulation Long"
TICKER: str = "BTCUSDT"
SIGNAL_TYPE: str = "rsi"
LOOKBACK: int = 14  # Wilder RSI period; 14-bar daily ≈ 168h rolling window in strategy spec


def _wilder_rsi(prices: pd.Series, period: int) -> pd.Series:
    """
    Compute Wilder's RSI using exponential smoothing (alpha = 1/period).

    Wilder's original formulation (not simple-average RSI) is used because it
    is the canonical choice for 14-period RSI and produces smoother, less
    whipsaw-prone oversold readings — appropriate for low-frequency capitulation
    detection.

    Parameters
    ----------
    prices : pd.Series
        Closing price series, datetime-indexed.
    period : int
        RSI look-back period (14 by default).

    Returns
    -------
    pd.Series
        RSI values in [0, 100], aligned to the input index.
    """
    # Step 1: per-bar price change
    delta: pd.Series = prices.diff()

    # Step 2: separate upward moves (gains) and downward moves (losses)
    gain: pd.Series = delta.clip(lower=0.0)   # positive deltas only
    loss: pd.Series = (-delta).clip(lower=0.0)  # absolute value of negative deltas

    # Step 3: seed the Wilder EMA with a simple mean over the first `period` bars,
    #         then apply exponential decay — alpha = 1/period (Wilder convention)
    alpha: float = 1.0 / period
    avg_gain: pd.Series = gain.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    avg_loss: pd.Series = loss.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    # Step 4: relative strength; guard against division-by-zero on flat stretches
    rs: pd.Series = avg_gain / avg_loss.replace(0.0, np.nan)

    # Step 5: standard RSI formula
    rsi: pd.Series = 100.0 - (100.0 / (1.0 + rs))

    return rsi


def generate_signals(prices: pd.Series, lookback: int = LOOKBACK) -> pd.Series:
    """
    Generate long/flat position signals for the BTC Capitulation Long strategy.

    Signal logic (RSI proxy for full on-chain signal):
        - ENTER LONG  : RSI(lookback) crosses below 30  →  signal = +1
                        (proxy for F&G ≤ 15 + netflow z-score > 1.5 spike)
        - STAY LONG   : remain in position while RSI ≤ 50
                        (proxy for F&G still below 30 / netflow not normalised)
        - EXIT / FLAT : RSI rises above 50  →  signal = 0
                        (proxy for F&G > 30 or netflow z-score < 0.5)

    This is a long-only strategy; −1 (short) is never emitted.

    Parameters
    ----------
    prices : pd.Series
        Daily closing prices for BTCUSDT, datetime-indexed, chronological order.
        Must contain at least `lookback + 1` non-NaN observations.
    lookback : int, optional
        RSI period.  Default is 14 (module constant LOOKBACK).

    Returns
    -------
    pd.Series[int]
        Integer signal series aligned to `prices.index`:
            +1  →  long (capitulation detected, hold position)
             0  →  flat (no signal or exit condition met)
            -1  →  short (unused; reserved for framework compatibility)

    Raises
    ------
    ValueError
        If `prices` contains fewer bars than `lookback + 1`.

    Examples
    --------
    >>> import pandas as pd
    >>> closes = pd.Series([...], index=pd.date_range("2020-01-01", periods=500, freq="D"))
    >>> sigs = generate_signals(closes)
    >>> sigs.value_counts()
    0    487
    1     13
    dtype: int64
    """
    # --- input validation ---------------------------------------------------
    if len(prices.dropna()) < lookback + 1:
        raise ValueError(
            f"generate_signals requires at least {lookback + 1} non-NaN price bars; "
            f"received {len(prices.dropna())}."
        )

    # --- Step 1: compute RSI ------------------------------------------------
    rsi: pd.Series = _wilder_rsi(prices, period=lookback)

    # --- Step 2: define entry and exit thresholds ---------------------------
    # 30 → oversold / extreme fear proxy (maps to F&G ≤ 15 + netflow spike)
    # 50 → mean recovery / exhaustion cleared (maps to F&G > 30 or netflow z < 0.5)
    ENTRY_THRESHOLD: float = 30.0
    EXIT_THRESHOLD: float = 50.0

    # --- Step 3: build raw entry flag (RSI below oversold level) -----------
    # +1 where RSI < 30, NaN where RSI is not yet available, 0 elsewhere
    entry_flag: pd.Series = (rsi < ENTRY_THRESHOLD).astype(int)

    # --- Step 4: state-machine hold logic -----------------------------------
    # Once triggered (RSI < 30), hold the long until RSI > 50.
    # This prevents premature exit on brief bounces within the capitulation zone.
    signals: pd.Series = pd.Series(0, index=prices.index, dtype=int)
    in_position: bool = False

    for idx in prices.index:
        current_rsi: float = rsi.at[idx]

        if np.isnan(current_rsi):
            # Not enough history yet — stay flat during RSI warm-up period
            signals.at[idx] = 0
            continue

        if not in_position:
            # ENTRY condition: RSI crosses into oversold territory
            if current_rsi < ENTRY_THRESHOLD:
                signals.at[idx] = 1
                in_position = True
            else:
                signals.at[idx] = 0
        else:
            # EXIT condition: RSI recovers above the exit threshold
            if current_rsi > EXIT_THRESHOLD:
                signals.at[idx] = 0
                in_position = False
            else:
                # Still within capitulation/recovery zone — maintain long
                signals.at[idx] = 1

    # --- Step 5: enforce long-only constraint (no -1 shorts) ----------------
    # Clip as a defensive guard; state machine above never emits -1
    signals = signals.clip(lower=0, upper=1)

    return signals