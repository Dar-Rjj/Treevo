import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Calculate the positive and negative part of the amount over volume
    df['positive_amount_vol'] = (df['amount'] / df['volume']).clip(lower=0)
    df['negative_amount_vol'] = (df['amount'] / df['volume']).clip(upper=0)

    # Volatility factor: Standard deviation of closing prices
    volatility = df['close'].rolling(window=5).std()

    # VWAP (Volume-Weighted Average Price)
    vwap = (df['amount'] / df['volume']).rolling(window=5).mean()

    # Momentum factor: 5-day return
    momentum = df['close'].pct_change(periods=5)

    # Mean reversion factor: Difference between current close and 5-day moving average
    mean_reversion = df['close'] - df['close'].rolling(window=5).mean()

    # Exponential smoothing of the mean reversion factor
    mean_reversion_smoothed = mean_reversion.ewm(span=5, adjust=False).mean()

    # Factor: ratio of positive sum to absolute negative sum
    pos_sum = df['positive_amount_vol'].rolling(window=5).sum()
    neg_sum_abs = df['negative_amount_vol'].abs().rolling(window=5).sum()
    liquidity_factor = pos_sum / (neg_sum_abs + 1e-7)

    # Advanced volatility measure: GARCH(1,1) volatility
    returns = df['close'].pct_change().dropna()
    squared_returns = returns**2
    omega = 1e-6
    alpha = 0.06
    beta = 0.92
    garch_volatility = [omega]
    for i in range(1, len(squared_returns)):
        garch_volatility.append(omega + alpha * squared_returns[i-1] + beta * garch_volatility[i-1])
    garch_volatility = pd.Series(garch_volatility, index=squared_returns.index)

    # Combine factors with dynamic weights using machine learning
    X = pd.concat([volatility, vwap, momentum, mean_reversion_smoothed, liquidity_factor, garch_volatility], axis=1).dropna()
    y = df['close'].pct_change().shift(-1).loc[X.index].dropna()

    model = LinearRegression()
    model.fit(X, y)
    dynamic_weights = model.coef_ / model.coef_.sum()  # Normalize coefficients to sum to 1

    combined_factor = (X * dynamic_weights).sum(axis=1)

    return combined_factor
