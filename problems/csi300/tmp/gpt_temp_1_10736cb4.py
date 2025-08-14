import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:

    # Calculate the positive and negative part of the amount over volume
    df['positive_amount_vol'] = (df['amount'] / df['volume']).clip(lower=0)
    df['negative_amount_vol'] = (df['amount'] / df['volume']).clip(upper=0)

    # Dynamic window for sum of positive and absolute negative parts
    pos_sum = df['positive_amount_vol'].rolling(window=df['volume'].rolling(window=5).mean().astype(int)).sum()
    neg_sum_abs = df['negative_amount_vol'].abs().rolling(window=df['volume'].rolling(window=5).mean().astype(int)).sum()

    # Factor: ratio of positive sum to absolute negative sum
    sentiment_factor = pos_sum / (neg_sum_abs + 1e-7)

    # Calculate the volatility using the close price with dynamic window
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    volatility = df['log_returns'].rolling(window=df['volume'].rolling(window=20).mean().astype(int)).std() * np.sqrt(252)

    # Calculate the VWAP with dynamic window
    vwap_window = df['volume'].rolling(window=20).mean().astype(int)
    df['vwap'] = (df['amount'] / df['volume']).rolling(window=vwap_window, min_periods=1).mean()

    # Exponential smoothing on the VWAP
    df['vwap_smoothed'] = df['vwap'].ewm(span=vwap_window, adjust=False).mean()

    # Momentum factor with dynamic window
    momentum = df['close'].pct_change(periods=df['volume'].rolling(window=20).mean().astype(int))

    # Mean reversion factor with dynamic window
    mean_reversion = -df['close'].pct_change(periods=df['volume'].rolling(window=5).mean().astype(int))

    # Trend strength factor: difference between 20-day and 5-day exponential moving averages with dynamic windows
    ema_20_window = df['volume'].rolling(window=20).mean().astype(int)
    ema_5_window = df['volume'].rolling(window=5).mean().astype(int)
    df['ema_20'] = df['close'].ewm(span=ema_20_window, adjust=False).mean()
    df['ema_5'] = df['close'].ewm(span=ema_5_window, adjust=False).mean()
    trend_strength = df['ema_5'] - df['ema_20']

    # Adaptive weighting based on recent market conditions
    recent_volatility = df['log_returns'].rolling(window=5).std() * np.sqrt(252)
    adaptive_weight_volatility = 1 / (1 + recent_volatility)

    # Combine factors with adaptive weights
    alpha_factor = (adaptive_weight_volatility * 0.3 * sentiment_factor + 
                    adaptive_weight_volatility * 0.2 * (1/volatility) + 
                    adaptive_weight_volatility * 0.2 * (df['vwap'] / df['vwap_smoothed']) + 
                    adaptive_weight_volatility * 0.15 * momentum + 
                    adaptive_weight_volatility * 0.15 * mean_reversion + 
                    adaptive_weight_volatility * 0.05 * trend_strength)

    return alpha_factor
