import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Calculate the positive and negative part of the amount over volume
    df['positive_amount_vol'] = (df['amount'] / df['volume']).clip(lower=0)
    df['negative_amount_vol'] = (df['amount'] / df['volume']).clip(upper=0)

    # Sum of positive and absolute negative parts with dynamic window size based on recent volatility
    recent_volatility = df['close'].pct_change().rolling(window=30).std()
    pos_sum = df['positive_amount_vol'].rolling(window=40 + 10 * recent_volatility).sum()
    neg_sum_abs = df['negative_amount_vol'].abs().rolling(window=40 + 10 * recent_volatility).sum()

    # Factor: ratio of positive sum to absolute negative sum
    sentiment_factor = pos_sum / (neg_sum_abs + 1e-7)

    # Calculate the log returns using the close price
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    volatility = df['log_returns'].rolling(window=60).std() * np.sqrt(252)

    # Calculate the VWAP
    df['vwap'] = (df['amount'] / df['volume']).rolling(window=60).mean()

    # Exponential smoothing on the VWAP
    df['vwap_smoothed'] = df['vwap'].ewm(span=60, adjust=False).mean()

    # Momentum factor
    momentum = df['close'].pct_change(periods=60)

    # Mean reversion factor
    mean_reversion = -df['close'].pct_change(periods=50)

    # Long-term trend strength factor: difference between 70-day and 50-day exponential moving averages
    df['ema_70'] = df['close'].ewm(span=70, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    long_trend_strength = df['ema_50'] - df['ema_70']

    # Adaptive weighting based on recent market conditions
    recent_volatility = df['log_returns'].rolling(window=30).std() * np.sqrt(252)
    adaptive_weight = 1 / (1 + np.exp(-recent_volatility))

    # Non-linear transformation of the factors
    sentiment_factor_nonlinear = np.sign(sentiment_factor) * np.sqrt(np.abs(sentiment_factor))
    vwap_ratio_nonlinear = np.sign(df['vwap'] / df['vwap_smoothed']) * np.sqrt(np.abs(df['vwap'] / df['vwap_smoothed']))

    # Additional factor: relative strength index (RSI)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    # Incorporate liquidity
    liquidity = df['volume'].rolling(window=30).mean()

    # Combine factors with adaptive weights
    alpha_factor = (
        0.3 * adaptive_weight * sentiment_factor_nonlinear +
        0.2 * (1 - adaptive_weight) * (1 / (volatility + 1e-7)) +
        0.15 * vwap_ratio_nonlinear +
        0.1 * momentum +
        0.07 * mean_reversion +
        0.06 * long_trend_strength +
        0.05 * ((rsi - 50) / 50) +  # Normalize RSI to be between -1 and 1
        0.07 * liquidity
    )

    return alpha_factor
