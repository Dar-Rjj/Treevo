import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Range
    df['intraday_range'] = df['high'] - df['low']
    
    # Calculate Price Change
    df['price_change'] = df['close'] - df['close'].shift(1)
    
    # Adjust Price Change by Intraday Range
    df['adjusted_price_change'] = df['price_change'] / df['intraday_range']
    
    # Weight by Volume
    df['weighted_adjusted_price_change'] = df['volume'] * df['adjusted_price_change']
    
    # Identify Volume Spikes
    df['vol_20_ma'] = df['volume'].rolling(window=20).mean()
    df['volume_spike'] = (df['volume'] > 1.5 * df['vol_20_ma']).astype(int)
    
    # Identify Price Spikes
    df['close_20_ma'] = df['close'].rolling(window=20).mean()
    df['price_spike'] = (df['close'] > 1.5 * df['close_20_ma']).astype(int)
    
    # Combine Momentum, Volume Spike, and Price Spike
    df['multiplier'] = 1
    df.loc[df['volume_spike'] & df['price_spike'], 'multiplier'] = 2.5
    df.loc[df['volume_spike'] & ~df['price_spike'], 'multiplier'] = 2
    df.loc[~df['volume_spike'] & df['price_spike'], 'multiplier'] = 1.5
    df['combined_momentum'] = df['weighted_adjusted_price_change'] * df['multiplier']
    
    # Calculate Cumulative Momentum over N days
    N = 5  # Example: Sum over the last 5 days
    df['cumulative_momentum'] = df['combined_momentum'].rolling(window=N).sum()
    
    return df['cumulative_momentum']
