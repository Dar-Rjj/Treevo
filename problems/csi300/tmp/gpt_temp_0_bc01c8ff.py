import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Close-to-Open Return
    df['close_to_open_return'] = df['open'].shift(-1) - df['close']
    
    # Volume Weighting
    df['volume_weighted_return'] = df['close_to_open_return'] * df['volume']
    
    # Determine Volatility
    df['volatility'] = df[['high', 'low', 'close']].rolling(window=30).std().mean(axis=1)
    
    # Adaptive Window Calculation
    def adjust_window_size(volatility, high_threshold, low_threshold, base_window):
        if volatility > high_threshold:
            return max(5, base_window - 10)  # Decrease window size, but not less than 5
        elif volatility < low_threshold:
            return base_window + 10  # Increase window size
        else:
            return base_window
    
    high_volatility_threshold = 0.05
    low_volatility_threshold = 0.01
    base_window_size = 30
    df['adaptive_window'] = df['volatility'].apply(lambda x: adjust_window_size(x, high_volatility_threshold, low_volatility_threshold, base_window_size))
    
    # Rolling Statistics with Adaptive Window
    df['rolling_mean'] = df.apply(lambda row: df.loc[:row.name, 'volume_weighted_return'].rolling(window=row['adaptive_window']).mean().iloc[-1], axis=1)
    df['rolling_std'] = df.apply(lambda row: df.loc[:row.name, 'volume_weighted_return'].rolling(window=row['adaptive_window']).std().iloc[-1], axis=1)
    
    # Recent Trend as a Momentum Indicator
    df['recent_trend'] = df['volume_weighted_return'] - df['rolling_mean']
    
    # Return the factor values
    return df['recent_trend']

# Example usage
# df = pd.read_csv('your_data.csv', index_col='date', parse_dates=True)
# factor_values = heuristics_v2(df)
# print(factor_values)
