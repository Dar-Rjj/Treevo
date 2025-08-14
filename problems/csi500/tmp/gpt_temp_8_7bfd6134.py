import pandas as pd
import pandas as pd

def heuristics_v2(df, n=10):
    # Compute High-Low Range
    df['high_low_range'] = df['high'] - df['low']
    
    # Compute Volume Adjusted High-Low Range
    df['volume_adjusted_high_low_range'] = df['high_low_range'] * df['volume']
    
    # Calculate Cumulative High-Low Range
    df['cumulative_high_low_range'] = df['volume_adjusted_high_low_range'].rolling(window=n).sum()
    
    # Calculate High-to-Low Price Ratio
    df['high_low_ratio'] = df['high'] / df['low']
    
    # Compute Volume Moving Average
    df['volume_ma'] = df['volume'].rolling(window=n).mean()
    
    # Determine Volume Trend
    df['volume_trend'] = (df['volume'] > df['volume_ma']).astype(int) * 2 - 1
    
    # Measure Volume Momentum
    df['volume_momentum'] = df['volume'].shift(1).rolling(window=n).sum() - df['volume'].rolling(window=n).sum()
    
    # Calculate Open-to-Close Price Change
    df['open_close_change'] = df['close'] - df['open']
    
    # Calculate Cumulative Open-to-Close Price Change
    df['cumulative_open_close_change'] = df['open_close_change'].rolling(window=n).sum()
    
    # Compute Average True Range
    df['true_range'] = df[['high', 'low']].apply(lambda x: max(x['high'], x.shift(1)['close']) - min(x['low'], x.shift(1)['close']), axis=1)
    df['average_true_range'] = df['true_range'].rolling(window=n).mean()
    
    # Combine Volume Trend with High-Low Price Ratio
    df['high_low_ratio_trend'] = df['high_low_ratio'] * df['volume_trend']
    
    # Incorporate Volume Momentum and Adjusted High-Low Range
    df['adjusted_high_low_range_momentum'] = df['volume_adjusted_high_low_range'] * df['volume_momentum']
    
    # Include Momentum and Range Expansion
    df['range_expansion'] = df['volume_adjusted_high_low_range'] - df['volume_adjusted_high_low_range'].shift(n-1)
    
    # Final Alpha Factor
    df['alpha_factor'] = df['high_low_ratio_trend'] + df['adjusted_high_low_range_momentum'] + df['range_expansion']
    
    return df['alpha_factor']

# Example usage
# df = pd.read_csv('stock_data.csv', parse_dates=['date'], index_col='date')
# alpha_factor = heuristics_v2(df)
