import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Calculate the positive and negative part of the amount over volume
    df['positive_amount_vol'] = (df['amount'] / df['volume']).clip(lower=0)
    df['negative_amount_vol'] = (df['amount'] / df['volume']).clip(upper=0)

    # Sum of positive and absolute negative parts
    pos_sum = df['positive_amount_vol'].rolling(window=10).sum()
    neg_sum_abs = df['negative_amount_vol'].abs().rolling(window=10).sum()

    # Factor: ratio of positive sum to absolute negative sum
    sentiment_factor = pos_sum / (neg_sum_abs + 1e-7)

    # Calculate the volatility using the close price
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    volatility = df['log_returns'].rolling(window=30).std() * np.sqrt(252)

    # Calculate the VWAP
    df['vwap'] = (df['amount'] / df['volume']).rolling(window=30).mean()

    # Exponential smoothing on the VWAP
    df['vwap_smoothed'] = df['vwap'].ewm(span=30, adjust=False).mean()

    # Momentum factor
    momentum = df['close'].pct_change(periods=30)

    # Mean reversion factor
    mean_reversion = -df['close'].pct_change(periods=10)

    # Long-term trend strength factor: difference between 60-day and 20-day exponential moving averages
    df['ema_60'] = df['close'].ewm(span=60, adjust=False).mean()
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    long_trend_strength = df['ema_20'] - df['ema_60']

    # Adaptive weighting based on recent market conditions
    recent_volatility = df['log_returns'].rolling(window=10).std() * np.sqrt(252)
    adaptive_weight = 1 / (1 + np.exp(-recent_volatility))

    # Dynamic window for sentiment factor
    dynamic_window = 10 + 5 * (recent_volatility > recent_volatility.quantile(0.75)).astype(int)
    pos_sum_dynamic = df['positive_amount_vol'].rolling(window=dynamic_window).sum()
    neg_sum_abs_dynamic = df['negative_amount_vol'].abs().rolling(window=dynamic_window).sum()
    sentiment_factor_dynamic = pos_sum_dynamic / (neg_sum_abs_dynamic + 1e-7)

    # Non-linear feature: squared log returns
    df['log_returns_squared'] = df['log_returns'] ** 2

    # Combine factors with adaptive weights
    alpha_factor = (
        0.3 * adaptive_weight * sentiment_factor_dynamic +
        0.2 * (1 - adaptive_weight) * (1 / volatility) +
        0.2 * (df['vwap'] / df['vwap_smoothed']) +
        0.15 * momentum +
        0.15 * mean_reversion +
        0.05 * long_trend_strength -
        0.05 * df['log_returns_squared'].rolling(window=30).mean()
    )

    return alpha_factor
