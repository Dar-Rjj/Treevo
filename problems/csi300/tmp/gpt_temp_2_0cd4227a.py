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
    df['true_range'] = df[['high', 'low', 'close']].apply(lambda x: max(x) - min(x), axis=1)
    volatility = df['true_range'].rolling(window=30).std()
    
    # Adaptive Window Calculation
    def adjust_window_size(volatility, window_size=60):
        if volatility > volatility.mean():
            return max(15, window_size - 10)  # Decrease window size if high volatility
        else:
            return min(120, window_size + 10)  # Increase window size if low volatility
    
    df['adaptive_window'] = df['true_range'].rolling(window=30).apply(
        lambda x: adjust_window_size(x), raw=False
    )
    
    # Rolling Statistics with Adaptive Window
    df['rolling_mean'] = df['volume_weighted_return'].rolling(
        window=df['adaptive_window'], min_periods=1
    ).mean()
    
    df['rolling_std'] = df['volume_weighted_return'].rolling(
        window=df['adaptive_window'], min_periods=1
    ).std()
    
    # Final Factor: Standardized Rolling Mean
    df['factor'] = (df['rolling_mean'] - df['rolling_mean'].mean()) / df['rolling_mean'].std()
    
    return df['factor']

# Example usage
# df = pd.read_csv('your_data.csv', parse_dates=['date'], index_col='date')
# factor = heuristics_v2(df)
# print(factor)
