import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Calculate the 5-day and 100-day Exponential Moving Averages (EMA) of the close price for smoothing
    ema_close_short = df['close'].ewm(span=5, adjust=False).mean().fillna(0)
    ema_close_long = df['close'].ewm(span=100, adjust=False).mean().fillna(0)

    # Calculate the Volume-Weighted Average Price (VWAP) over a dynamic window size based on average volume
    vwap_window = df['volume'].rolling(window=5).mean().astype(int)
    vwap = (df['amount'] * df['close']).rolling(window=vwap_window).sum() / df['volume'].rolling(window=vwap_window).sum()
    vwap.fillna(0, inplace=True)

    # Calculate the difference between today's EMA and yesterday's EMA, then scale by the 5-day average of the amount
    momentum_scaled_ema_short = (ema_close_short.diff() / df['amount'].rolling(window=5).mean()).fillna(0)

    # Compute the relative range of today's trading compared to the last 5 days
    relative_range = (df['high'] - df['low']) / df['close'].shift(1)
    relative_range_rank = relative_range.rolling(window=5).rank(pct=True).fillna(0)

    # Calculate the relative strength as the ratio of the current EMA to the 5-day minimum EMA
    relative_strength = ema_close_short / ema_close_short.rolling(window=5).min().fillna(0)

    # Calculate the 5-day rolling standard deviation of the VWAP as a measure of volatility
    volatility_vwap = vwap.rolling(window=5).std().fillna(0)

    # Integrate macroeconomic indicators (assuming they are available in the DataFrame)
    if 'macro_indicator' in df.columns:
        macro_indicator = df['macro_indicator']
        macro_adjustment = (macro_indicator - macro_indicator.rolling(window=30).mean()) / macro_indicator.rolling(window=30).std()
    else:
        macro_adjustment = 1  # No adjustment if no macroeconomic data is available

    # Combine the factors: momentum scaled EMA, relative range rank, and relative strength, adjusted by VWAP volatility and macroeconomic indicator
    factor = (momentum_scaled_ema_short * relative_range_rank * relative_strength) / (volatility_vwap + 1e-6) * macro_adjustment

    # Adjust for liquidity by scaling the factor with the 100-day EMA
    factor_liquidity_adjusted = factor * (ema_close_long / df['close']).fillna(0)

    return factor_liquidity_adjusted
