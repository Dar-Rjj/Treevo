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
    df['combined_momentum'] = df['high_low_momentum'] * df['volume_change']
    df['combined_momentum'] = df['combined_momentum'].apply(lambda x: x if df['volume_change'] > 0 else -x)
    
    # Identify Volume Spike Days
    df['volume_spike'] = df['volume'] > df['volume'].rolling(window=20).mean()
    
    # Adjust 10-day Exponential Moving Average
    df['adjusted_ema_10_returns'] = df['ema_10_returns'] * (0.5 if df['volume_spike'] else 1)
    
    # Final Alpha Factor
    df['alpha_factor'] = df['combined_momentum'] + df['adjusted_ema_10_returns']
    
    return df['alpha_factor']
