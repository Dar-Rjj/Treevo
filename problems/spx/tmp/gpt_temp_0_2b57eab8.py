import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Return
    df['intraday_return'] = df['close'] - df['open']
    
    # Calculate Adjusted High-Low Range
    df['adjusted_high_low_range'] = (df['high'] - df['low']) / df['open']
    
    # Combine Intraday Return and Adjusted High-Low Range
    df['combined_factor'] = df['intraday_return'] * df['adjusted_high_low_range']
    
    # Volume Weighting
    df['volume_weighted_factor'] = df['combined_factor'] * df['volume']
    
    # Calculate Daily Momentum
    df['daily_momentum'] = df['close'].diff()
    
    # Calculate Price Momentum
    df['sma_10'] = df['close'].rolling(window=10).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['price_momentum'] = df['sma_10'] - df['sma_50']
    
    # Confirm Momentum with Volume
    df['cumulative_short_term_volume'] = df['volume'].rolling(window=10).sum()
    df['cumulative_long_term_volume'] = df['volume'].rolling(window=50).sum()
    df['volume_difference'] = df['cumulative_short_term_volume'] - df['cumulative_long_term_volume']
    df['confirmed_momentum'] = df['price_momentum'] * df['volume_difference']
    
    # Adjusted Close-to-Open Return by Volume
    df['close_to_open_return'] = (df['close'] - df['open']) * df['volume']
    df['volume_sum_past_5_days'] = df['volume'].rolling(window=5).sum()
    df['adjusted_close_to_open_return'] = df['close_to_open_return'] / df['volume_sum_past_5_days']
    
    # Detect Volume Spike
    df['average_volume_5_days'] = df['volume'].rolling(window=5).mean()
    df['volume_spike'] = df['volume'] > df['average_volume_5_days']
    
    # Final Alpha Factor
    df['final_alpha_factor'] = df['daily_momentum'] + df['volume_weighted_factor'] + df['adjusted_close_to_open_return']
    
    # Weight by Inverse of Volume Spike Factor and Average Volume
    df['volume_spike_factor'] = df['volume_spike'].apply(lambda x: 1 if not x else (df['average_volume_5_days'] / df['volume']))
    df['final_alpha_factor'] = df['final_alpha_factor'] * df['volume_spike_factor']
    
    return df['final_alpha_factor']

# Example usage:
# df = pd.read_csv('your_data.csv', index_col='date', parse_dates=True)
# alpha_factor = heuristics_v2(df)
# print(alpha_factor)
