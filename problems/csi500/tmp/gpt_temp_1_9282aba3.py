import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Adaptive momentum with exponential smoothing, considering both short and long-term trends
    short_term_momentum = df['close'].pct_change(periods=5).ewm(span=5, adjust=False).mean()
    long_term_momentum = df['close'].pct_change(periods=30).ewm(span=30, adjust=False).mean()
    adaptive_momentum = 0.7 * short_term_momentum + 0.3 * long_term_momentum

    # Dynamic volatility using the standard deviation of logarithmic returns over the last 14 days, with additional smoothing
    log_returns = np.log(df['close'] / df['close'].shift(1))
    dynamic_volatility = log_returns.rolling(window=14).std().ewm(span=14, adjust=False).mean()

    # Volume trend analysis using the change in volume over a 10-day period
    volume_trend = df['volume'].pct_change(periods=10).ewm(span=10, adjust=False).mean()

    # Market sentiment using the ratio of high to low prices as a proxy, combined with the average trading amount
    market_sentiment = ((df['high'] - df['low']) / df['close']) * (df['amount'] / df['amount'].mean())

    # Seasonality factor: simple day-of-week effect
    seasonality_effect = df.index.dayofweek.map({0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4, 4: 0.5, 5: 0.6, 6: 0.7})

    # Combining the factors into a single alpha factor
    # Weights are adjusted based on their perceived importance
    factor_values = (
        0.35 * adaptive_momentum
        - 0.25 * dynamic_volatility
        + 0.2 * volume_trend
        + 0.15 * market_sentiment
        + 0.05 * seasonality_effect
    )

    return factor_values
