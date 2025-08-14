import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame, sector_trends: pd.Series, macro_indicators: pd.DataFrame) -> pd.Series:
    # Adaptive momentum with exponential smoothing, considering both short and long-term trends
    short_term_momentum = df['close'].pct_change(periods=5).ewm(span=5, adjust=False).mean()
    long_term_momentum = df['close'].pct_change(periods=30).ewm(span=30, adjust=False).mean()
    adaptive_momentum = 0.7 * short_term_momentum + 0.3 * long_term_momentum

    # Dynamic volatility using the standard deviation of logarithmic returns over the last 14 days
    log_returns = np.log(df['close'] / df['close'].shift(1))
    dynamic_volatility = log_returns.rolling(window=14).std().ewm(span=14, adjust=False).mean()

    # Volume trend indicator to capture liquidity conditions
    volume_trend = df['volume'].pct_change().ewm(span=10, adjust=False).mean()

    # Market sentiment using the ratio of high to low prices as a proxy
    market_sentiment = (df['high'] - df['low']) / df['close']

    # Advanced smoothing and cumulative trends
    cumulative_trend = df['close'].pct_change().cumsum()

    # Sector-specific trends
    sector_trend_adjustment = sector_trends.ewm(span=20, adjust=False).mean()

    # Macroeconomic indicators
    macro_trend = macro_indicators.mean(axis=1).ewm(span=20, adjust=False).mean()

    # Adjusting weights based on market conditions and sector-specific trends
    volatility_adjustment = np.clip(dynamic_volatility, 0.05, 0.5)
    momentum_adjustment = np.clip(adaptive_momentum, -0.5, 0.5)

    # Machine learning for dynamic weight adjustments
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    X = pd.concat([adaptive_momentum, dynamic_volatility, volume_trend, market_sentiment, cumulative_trend, sector_trend_adjustment, macro_trend], axis=1).dropna()
    y = df['close'].pct_change().loc[X.index]
    model.fit(X, y)
    weights = model.coef_

    # Combining the factors into a single alpha factor with ML-adjusted weights
    factor_values = (
        weights[0] * adaptive_momentum 
        - weights[1] * dynamic_volatility 
        + weights[2] * volume_trend 
        + weights[3] * market_sentiment 
        + weights[4] * cumulative_trend 
        + weights[5] * sector_trend_adjustment 
        + weights[6] * macro_trend
    )

    return factor_values
