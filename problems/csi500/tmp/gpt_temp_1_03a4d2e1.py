import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Adaptive momentum considering both short and long-term trends with exponential smoothing
    short_term_momentum = df['close'].pct_change(periods=5).ewm(span=5, adjust=False).mean()
    long_term_momentum = df['close'].pct_change(periods=30).ewm(span=30, adjust=False).mean()
    adaptive_momentum = 0.6 * short_term_momentum + 0.4 * long_term_momentum

    # Dynamic volatility using the standard deviation of logarithmic returns over the last 14 days
    log_returns = np.log(df['close'] / df['close'].shift(1))
    dynamic_volatility = log_returns.rolling(window=14).std().ewm(span=14, adjust=False).mean()

    # Normalized volume to account for varying liquidity conditions
    normalized_volume = (df['volume'] - df['volume'].min()) / (df['volume'].max() - df['volume'].min())

    # Market sentiment using the ratio of high to low prices as a proxy
    market_sentiment = (df['high'] - df['low']) / df['close']

    # Cumulative trend using the cumulative sum of daily returns
    cumulative_trend = (df['close'] / df['close'].iloc[0] - 1).cumsum()

    # Sector-specific insights using the relative strength compared to the sector index (assuming 'sector_index' is available in the DataFrame)
    relative_strength = df['close'] / df['sector_index']

    # Combining the six factors into a single alpha factor
    # Weights are adjusted based on their perceived importance
    factor_values = (0.3 * adaptive_momentum - 0.15 * dynamic_volatility + 0.2 * normalized_volume + 
                     0.1 * market_sentiment + 0.15 * cumulative_trend + 0.1 * relative_strength)

    return factor_values
