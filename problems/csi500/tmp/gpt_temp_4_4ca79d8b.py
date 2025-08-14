import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate High-Low Range
    df['high_low_range'] = df['high'] - df['low']
    
    # Calculate High-Low Range Momentum
    df['high_low_momentum'] = df['high_low_range'] - df['high_low_range'].shift(1)
    
    # Calculate Daily Returns
    df['daily_returns'] = df['close'] / df['close'].shift(1) - 1
    
    # Calculate 10-day Exponential Moving Average of Returns
    df['ema_10_returns'] = df['daily_returns'].ewm(span=10, adjust=False).mean()
    
    # Calculate Volume Change
    df['volume_change'] = df['volume'] - df['volume'].shift(1)
    
    # Combine High-Low Range Momentum and Volume Change
    df['combined_factor'] = df['high_low_momentum'] * df['volume_change']
    
    # Adjust for Sign
    df['combined_factor'] = df['combined_factor'] * (df['volume_change'] > 0) - df['combined_factor'] * (df['volume_change'] < 0)
    
    # Identify Volume Spike Days
    df['volume_moving_avg_20'] = df['volume'].rolling(window=20).mean()
    df['is_volume_spike'] = df['volume'] > df['volume_moving_avg_20'] * 2
    
    # Adjust 10-day Exponential Moving Average
    df['adjusted_ema_10_returns'] = df['ema_10_returns'] * (0.5 if df['is_volume_spike'] else 1)
    
    # Final Alpha Factor
    df['alpha_factor'] = df['combined_factor'] + df['adjusted_ema_10_returns']
    
    return df['alpha_factor']
