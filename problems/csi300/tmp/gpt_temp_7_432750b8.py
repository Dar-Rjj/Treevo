import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Close-to-Open Return
    df['close_to_open_return'] = df['open'].shift(-1) - df['close']
    
    # Volume Weighting
    df['volume_weighted_return'] = df['close_to_open_return'] * df['volume']
    
    # Determine Volatility using High, Low, and Close prices
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    fixed_window = 20  # Fixed window for initial volatility calculation
    df['volatility'] = df['typical_price'].rolling(window=fixed_window).std()
    
    # Adaptive Window Size based on Volatility
    def adjust_window_size(vol):
        if vol > df['volatility'].mean():
            return max(5, int(fixed_window * 0.8))  # Decrease window size
        else:
            return min(40, int(fixed_window * 1.2))  # Increase window size
    
    # Apply the adaptive window size
    df['window_size'] = df['volatility'].apply(adjust_window_size)
    
    # Rolling Statistics
    df['rolling_mean'] = df['volume_weighted_return'].rolling(window=df['window_size']).mean()
    df['rolling_std'] = df['volume_weighted_return'].rolling(window=df['window_size']).std()
    
    # Adjust Alpha Factor using Z-Score
    df['alpha_factor'] = (df['volume_weighted_return'] - df['rolling_mean']) / df['rolling_std']
    
    # Drop NaN values
    df.dropna(inplace=True)
    
    return df['alpha_factor']

# Example usage
# df = pd.read_csv('your_data.csv', index_col='date', parse_dates=True)
# alpha_factor = heuristics_v2(df)
