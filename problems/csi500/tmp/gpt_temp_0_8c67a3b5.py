import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Adaptive momentum considering both short and long-term trends with exponential smoothing
    short_term_momentum = df['close'].pct_change(periods=5).ewm(span=5, adjust=False).mean()
    long_term_momentum = df['close'].pct_change(periods=30).ewm(span=30, adjust=False).mean()
    adaptive_momentum = 0.6 * short_term_momentum + 0.4 * long_term_momentum

    # Dynamic volatility using the standard deviation of logarithmic returns over the last 21 days
    log_returns = np.log(df['close'] / df['close'].shift(1))
    dynamic_volatility = log_returns.rolling(window=21).std().ewm(span=21, adjust=False).mean()

    # Volume trend using the percentage change in volume to capture liquidity shifts
    volume_trend = df['volume'].pct_change(periods=5).ewm(span=5, adjust=False).mean()

    # Market sentiment using the ratio of close to open prices as a proxy
    market_sentiment = (df['close'] - df['open']) / df['open']

    # Sector-specific trend by calculating the average return of the sector
    # Assuming 'sector' is a column in the DataFrame
    sector_trend = df.groupby('sector')['close'].pct_change(periods=30).transform('mean')

    # Combining the factors into a single alpha factor
    # Weights are adjusted based on their perceived importance
    factor_values = 0.3 * adaptive_momentum - 0.2 * dynamic_volatility + 0.2 * volume_trend + 0.1 * market_sentiment + 0.2 * sector_trend

    return factor_values
