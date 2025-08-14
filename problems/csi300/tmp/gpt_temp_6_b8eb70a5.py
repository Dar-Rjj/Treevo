import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def heuristics_v2(df: pd.DataFrame, macroeconomic_data: pd.DataFrame) -> pd.Series:
    # Calculate the positive and negative part of the amount over volume
    df['positive_amount_vol'] = (df['amount'] / df['volume']).clip(lower=0)
    df['negative_amount_vol'] = (df['amount'] / df['volume']).clip(upper=0)

    # Sum of positive and absolute negative parts
    pos_sum = df['positive_amount_vol'].rolling(window=5).sum()
    neg_sum_abs = df['negative_amount_vol'].abs().rolling(window=5).sum()

    # Factor: ratio of positive sum to absolute negative sum
    sentiment_factor = pos_sum / (neg_sum_abs + 1e-7)

    # Calculate the log returns for volatility
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    volatility = df['log_returns'].rolling(window=20).std() * np.sqrt(252)

    # Calculate the VWAP
    df['vwap'] = (df['amount'] / df['volume']).rolling(window=20).mean()

    # Exponential smoothing on the VWAP
    df['vwap_smoothed'] = df['vwap'].ewm(span=20, adjust=False).mean()

    # Momentum factor
    momentum = df['close'].pct_change(periods=20)

    # Mean reversion factor
    mean_reversion = -df['close'].pct_change(periods=5)

    # Long-term trend strength factor: difference between 60-day and 20-day exponential moving averages
    df['ema_60'] = df['close'].ewm(span=60, adjust=False).mean()
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    long_trend_strength = df['ema_20'] - df['ema_60']

    # Adaptive weighting based on recent market conditions
    recent_volatility = df['log_returns'].rolling(window=5).std() * np.sqrt(252)
    adaptive_weight = 1 / (1 + np.exp(-recent_volatility))

    # Non-linear transformations
    vwap_ratio_squared = (df['vwap'] / df['vwap_smoothed']) ** 2
    momentum_squared = momentum ** 2
    mean_reversion_squared = mean_reversion ** 2

    # Integrate macroeconomic indicators
    df = df.join(macroeconomic_data, how='left')
    macro_features = ['gdp_growth', 'inflation_rate', 'unemployment_rate']
    df[macro_features] = df[macro_features].fillna(method='ffill')

    # Use machine learning for dynamic weighting
    X = df[['volatility', 'vwap_ratio_squared', 'momentum_squared', 'mean_reversion_squared', 'long_trend_strength'] + macro_features].dropna()
    y = df['close'].pct_change(periods=20).shift(-20).dropna()
    common_index = X.index.intersection(y.index)
    X = X.loc[common_index]
    y = y.loc[common_index]

    model = LinearRegression()
    model.fit(X, y)
    weights = model.coef_

    # Combine factors with learned weights
    alpha_factor = (
        sentiment_factor * weights[0] +
        (1 / volatility) * weights[1] +
        vwap_ratio_squared * weights[2] +
        momentum_squared * weights[3] +
        mean_reversion_squared * weights[4] +
        long_trend_strength * weights[5] +
        df['gdp_growth'] * weights[6] +
        df['inflation_rate'] * weights[7] +
        df['unemployment_rate'] * weights[8]
    )

    return alpha_factor
