import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:

    # Calculate the positive and negative part of the amount over volume
    df['positive_amount_vol'] = (df['amount'] / df['volume']).clip(lower=0)
    df['negative_amount_vol'] = (df['amount'] / df['volume']).clip(upper=0)

    # Sum of positive and absolute negative parts with dynamic window
    pos_sum = df['positive_amount_vol'].rolling(window=df['volume'].rolling(window=5).mean().astype(int)).sum()
    neg_sum_abs = df['negative_amount_vol'].abs().rolling(window=df['volume'].rolling(window=5).mean().astype(int)).sum()

    # Factor: ratio of positive sum to absolute negative sum
    sentiment_factor = pos_sum / (neg_sum_abs + 1e-7)

    # Calculate the volatility using the close price
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    volatility = df['log_returns'].rolling(window=20).std() * np.sqrt(252)

    # Calculate the VWAP
    df['vwap'] = (df['amount'] / df['volume']).rolling(window=20).mean()

    # Exponential smoothing on the VWAP
    df['vwap_smoothed'] = df['vwap'].ewm(span=20, adjust=False).mean()

    # Momentum factor with adaptive window
    momentum_window = df['close'].rolling(window=5).kurt().apply(lambda x: 20 if x < 3 else 5)
    momentum = df['close'].pct_change(periods=momentum_window)

    # Mean reversion factor with adaptive window
    mean_reversion_window = df['close'].rolling(window=5).skew().apply(lambda x: 5 if x > 0.5 else 2)
    mean_reversion = -df['close'].pct_change(periods=mean_reversion_window)

    # Trend strength factor: difference between 20-day and 5-day exponential moving averages
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
    trend_strength = df['ema_5'] - df['ema_20']

    # Adaptive weighting based on recent market conditions
    recent_volatility = df['log_returns'].rolling(window=5).std() * np.sqrt(252)
    adaptive_weight = 1 / (1 + np.exp(-recent_volatility))

    # Sector-specific factors (assuming sector information is available in a column 'sector')
    sector_mean_close = df.groupby('sector')['close'].transform('mean')
    sector_factor = (df['close'] - sector_mean_close) / sector_mean_close

    # Combine factors with adaptive weights
    alpha_factor = (adaptive_weight * sentiment_factor + 
                    (1 - adaptive_weight) * (1/volatility) + 
                    0.2 * (df['vwap'] / df['vwap_smoothed']) + 
                    0.15 * momentum + 
                    0.15 * mean_reversion + 
                    0.05 * trend_strength + 
                    0.05 * sector_factor)

    return alpha_factor
