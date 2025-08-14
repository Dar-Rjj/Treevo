import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Daily Close-to-Close Return
    df['daily_return'] = df['close'].pct_change()

    # Apply Exponential Moving Average (EMA) to Returns
    lookback_period = 20
    df['ema_returns'] = df['daily_return'].ewm(span=lookback_period, adjust=False).mean()

    # Compute EMA of Squared Returns for Volatility
    df['squared_returns'] = df['daily_return'] ** 2
    df['ema_volatility'] = df['squared_returns'].ewm(span=lookback_period, adjust=False).mean().sqrt()

    # Calculate Divergence between EMA of Returns and Volatility
    df['divergence'] = (df['ema_volatility'] - df['ema_returns']).abs()

    # Weight Divergence by Volume and High-Low Range
    df['high_low_range'] = df['high'] - df['low']
    df['weighted_divergence'] = (df['divergence'] * df['volume']) / df['high_low_range']

    # Calculate Intraday Volatility
    df['intraday_volatility'] = (df['high'] - df['low']).rolling(window=lookback_period).mean()

    # Calculate Intraday Momentum
    df['intraday_return'] = (df['high'] - df['low']) / (df['close'] - df['open'])

    # Calculate Volume Weighted Price
    df['volume_weighted_price'] = (df['close'] * df['volume'] + df['open'] * df['volume']) / (df['volume'] * 2)

    # Calculate Volume Weighted Price Change
    df['volume_weighted_price_change'] = df['volume_weighted_price'].diff()

    # Compute Combined Intraday and Daily Momentum
    df['combined_momentum'] = df['intraday_return'] + df['volume_weighted_price_change']

    # Integrate Momentum and Reversal
    df['integrated_momentum'] = df['combined_momentum'] - df['combined_momentum'].ewm(span=30, adjust=False).mean()

    # Combine Initial Factors
    df['raw_combined_factor'] = df['intraday_volatility'] * df['integrated_momentum']

    # Adjust for Trading Volume
    k_days = 10
    df['average_volume'] = df['volume'].rolling(window=k_days).mean()
    df['adjusted_combined_factor'] = df['raw_combined_factor'] / df['average_volume']

    # Calculate Volume Thrust
    df['volume_ratio'] = df['volume'] / df['volume'].shift(1)
    df['normalized_intraday_range'] = (df['high'] - df['low']) / df['close']
    df['volume_thrust'] = df['volume_ratio'] * df['normalized_intraday_range']

    # Final Combination and Smoothing
    df['final_factor'] = df['adjusted_combined_factor'] * df['weighted_divergence'] * df['volume_thrust']
    window_size = 5
    df['smoothed_factor'] = df['final_factor'].rolling(window=window_size).mean()

    return df['smoothed_factor']
