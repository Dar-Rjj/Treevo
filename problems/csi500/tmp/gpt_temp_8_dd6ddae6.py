import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Adaptive momentum considering both short and long-term trends with exponential smoothing
    short_term_momentum = df['close'].pct_change(periods=5).ewm(span=5, adjust=False).mean()
    long_term_momentum = df['close'].pct_change(periods=30).ewm(span=30, adjust=False).mean()
    adaptive_momentum = 0.7 * short_term_momentum + 0.3 * long_term_momentum

    # Dynamic volatility using the standard deviation of logarithmic returns over the last 14 days
    log_returns = np.log(df['close'] / df['close'].shift(1))
    dynamic_volatility = log_returns.rolling(window=14).std().ewm(span=14, adjust=False).mean()

    # Normalized volume to account for varying liquidity conditions
    normalized_volume = (df['volume'] - df['volume'].mean()) / df['volume'].std()

    # Market sentiment using the ratio of high to low prices as a proxy
    market_sentiment = (df['high'] - df['low']) / df['close']

    # Incorporate macroeconomic indicators
    macro_indicator = df['amount'].pct_change(periods=1).rolling(window=10).mean()

    # Non-linear factor combinations
    non_linear_factor = (adaptive_momentum ** 2) - (dynamic_volatility ** 2) + (normalized_volume ** 2) + (market_sentiment ** 2)

    # Use machine learning for dynamic weighting
    from sklearn.linear_model import LinearRegression
    X = pd.DataFrame({
        'adaptive_momentum': adaptive_momentum,
        'dynamic_volatility': dynamic_volatility,
        'normalized_volume': normalized_volume,
        'market_sentiment': market_sentiment,
        'macro_indicator': macro_indicator
    })
    y = log_returns.shift(-1).dropna()  # Predicting next day's return
    X = X.dropna().iloc[:-1]  # Aligning with y
    model = LinearRegression()
    model.fit(X, y)
    weights = model.coef_

    # Combining the factors into a single alpha factor with machine-learned weights
    factor_values = (
        weights[0] * adaptive_momentum + 
        weights[1] * dynamic_volatility + 
        weights[2] * normalized_volume + 
        weights[3] * market_sentiment + 
        weights[4] * macro_indicator
    )

    return factor_values
