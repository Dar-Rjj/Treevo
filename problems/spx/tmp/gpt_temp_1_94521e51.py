import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Return
    df['intraday_return'] = df['close'] - df['open']
    
    # Calculate High-Low Range
    df['high_low_range'] = df['high'] - df['low']
    
    # Adjust by Open Price
    df['adjusted_high_low_range'] = df['high_low_range'] / df['open']
    
    # Combine Intraday Return and Adjusted High-Low Range
    df['combined_intraday_adjusted'] = df['intraday_return'] * df['adjusted_high_low_range']
    
    # Volume Weighting
    df['volume_weighted_combined'] = df['combined_intraday_adjusted'] * df['volume']
    
    # Calculate Daily Momentum
    df['daily_momentum'] = df['close'].diff()
    
    # Calculate Short-Term and Long-Term SMA
    df['sma_10'] = df['close'].rolling(window=10).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    
    # Calculate Price Momentum
    df['price_momentum'] = df['sma_10'] - df['sma_50']
    
    # Compute Cumulative Volume
    df['cum_volume_10'] = df['volume'].rolling(window=10).sum()
    df['cum_volume_50'] = df['volume'].rolling(window=50).sum()
    
    # Calculate Volume Difference
    df['volume_difference'] = df['cum_volume_10'] - df['cum_volume_50']
    
    # Combine Momentum and Volume
    df['momentum_volume_combined'] = df['price_momentum'] * df['volume_difference']
    
    # Adjusted Close-to-Open Return by Volume
    df['close_to_open_return'] = df['close'] - df['open']
    df['total_volume_5_days'] = df['volume'].rolling(window=5).sum()
    df['adjusted_close_to_open_return'] = (df['close_to_open_return'] * df['volume']) / df['total_volume_5_days']
    
    # Sum of High-Low Spread and Adjusted Close-to-Open Return
    df['sum_high_low_spread_adjusted_return'] = df['high_low_range'] + df['adjusted_close_to_open_return']
    
    # Detect Volume Spike
    df['average_volume_5_days'] = df['volume'].rolling(window=5).mean()
    df['volume_spike_factor'] = (df['volume'] > df['average_volume_5_days']).map(lambda x: 1 if not x else 1 / (df['volume'] / df['average_volume_5_days']))
    
    # Final Alpha Factor
    df['alpha_factor'] = (df['daily_momentum'] + df['volume_weighted_combined'] + df['adjusted_close_to_open_return']) * df['volume_spike_factor']
    
    return df['alpha_factor']
