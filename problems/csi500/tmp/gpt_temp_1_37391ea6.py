import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate High-Low Range
    df['high_low_range'] = df['high'] - df['low']
    
    # Calculate High-Low Range Momentum
    df['high_low_range_momentum'] = df['high_low_range'] - df['high_low_range'].shift(1)
    
    # Calculate Daily Returns
    df['daily_returns'] = (df['close'] / df['close'].shift(1)) - 1
    
    # Calculate 10-day Moving Average of Returns
    df['10_day_ma_returns'] = df['daily_returns'].rolling(window=10).mean()
    
    # Calculate Volume Change
    df['volume_change'] = df['volume'] - df['volume'].shift(1)
    
    # Combine High-Low Range Momentum and Volume Change
    df['combined_factor'] = df['high_low_range_momentum'] * df['volume_change']
    
    # Adjust for Sign
    df['combined_factor'] = df['combined_factor'].apply(lambda x: x if df['volume_change'] > 0 else -x)
    
    # Identify Volume Spike Days
    df['20_day_ma_volume'] = df['volume'].rolling(window=20).mean()
    df['volume_spike'] = df['volume'] > df['20_day_ma_volume'] * 2
    
    # Adjust 10-day Moving Average for Volume Spikes
    df['adjusted_10_day_ma_returns'] = df.apply(
        lambda row: row['10_day_ma_returns'] * 0.5 if row['volume_spike'] else row['10_day_ma_returns'], axis=1
    )
    
    # Add Adjusted 10-day Moving Average to Combined High-Low Range Momentum
    df['final_alpha_factor'] = df['combined_factor'] + df['adjusted_10_day_ma_returns']
    
    return df['final_alpha_factor']
