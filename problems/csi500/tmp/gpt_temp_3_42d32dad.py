import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Adaptive momentum with exponential smoothing for short, medium, and long-term trends
    short_term_momentum = df['close'].pct_change(periods=5).ewm(span=5, adjust=False).mean()
    medium_term_momentum = df['close'].pct_change(periods=20).ewm(span=20, adjust=False).mean()
    long_term_momentum = df['close'].pct_change(periods=60).ewm(span=60, adjust=False).mean()
    adaptive_momentum = 0.4 * short_term_momentum + 0.3 * medium_term_momentum + 0.3 * long_term_momentum

    # Dynamic volatility using the standard deviation of logarithmic returns over the last 21 days
    log_returns = np.log(df['close'] / df['close'].shift(1))
    dynamic_volatility = log_returns.rolling(window=21).std().ewm(span=21, adjust=False).mean()

    # Volume-weighted average price (VWAP) to capture intraday trading dynamics
    vwap = (df['amount'] / df['volume']).rolling(window=21).mean()

    # Market sentiment using the ratio of high to low prices as a proxy
    market_sentiment = (df['high'] - df['low']) / df['close']

    # Sector-specific trend analysis using a simple moving average crossover
    short_sma = df['close'].rolling(window=10).mean()
    long_sma = df['close'].rolling(window=30).mean()
    sma_crossover = short_sma - long_sma

    # Incorporate seasonality
    df['month'] = df.index.month
    seasonal_effect = df.groupby('month')['close'].transform(lambda x: (x - x.mean()) / x.std())

    # Incorporate external events (example: news sentiment score)
    # Assuming 'news_sentiment' is a column in the DataFrame representing daily news sentiment scores
    if 'news_sentiment' in df.columns:
        news_impact = df['news_sentiment'].rolling(window=5).mean()
    else:
        news_impact = pd.Series(0, index=df.index)

    # Machine learning for adaptive weights
    # Example: Using a simple linear regression model to determine the weights
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler

    X = pd.concat([adaptive_momentum, dynamic_volatility, vwap, market_sentiment, sma_crossover, seasonal_effect, news_impact], axis=1).dropna()
    y = df['close'].pct_change().shift(-1).loc[X.index]  # Target: next day's return
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LinearRegression()
    model.fit(X_scaled, y)
    weights = model.coef_

    # Combine the factors into a single alpha factor using adaptive weights
    factor_values = (X * weights).sum(axis=1)

    return factor_values
