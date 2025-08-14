import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate 30-day Price Momentum
    df['30_day_momentum'] = (df['close'] - df['close'].shift(30)) / df['close'].shift(30)
    
    # Identify Volume Shock
    df['30_day_avg_volume'] = df['volume'].rolling(window=30, min_periods=1).mean()
    df['volume_shock'] = (df['volume'] > 2 * df['30_day_avg_volume']).astype(int)
    
    # Multiply 30-day Momentum by Volume Shock indicator
    df['momentum_volume_shock'] = df['30_day_momentum'] * df['volume_shock']
    
    # Calculate High-to-Low Range
    df['high_low_range'] = df['high'] - df['low']
    
    # Adjust for Volume
    df['adjusted_high_low_range'] = df['high_low_range'] * df['volume']
    
    # Detect Volume Spike
    df['5_day_avg_volume'] = df['volume'].rolling(window=5, min_periods=1).mean()
    df['volume_spike'] = (df['volume'] > 1.7 * df['5_day_avg_volume']).astype(int)
    
    # Calculate 10-day Momentum
    df['10_day_momentum'] = df['close'] - df['close'].shift(10)
    
    # Combine Adjusted Range, 10-day Momentum, and Volume Spike
    df['combined_factor'] = (df['adjusted_high_low_range'] + df['10_day_momentum']) * (2.5 if df['volume_spike'] == 1 else 1)
    
    # High-Low Range Momentum
    N = 10
    df['high_low_range_momentum'] = df['high_low_range'] - df['high_low_range'].shift(N)
    
    # Synthesize Final Alpha Factor
    df['alpha_factor'] = df['momentum_volume_shock'] + df['combined_factor'] + df['high_low_range_momentum']
    
    return df['alpha_factor']
