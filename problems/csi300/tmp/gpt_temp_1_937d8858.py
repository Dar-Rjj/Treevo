import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Close-to-Open Return
    df['close_to_open_return'] = df['open'].shift(-1) - df['close']
    
    # Enhanced Volume Weighting
    df['volume_ema'] = df['volume'].ewm(span=20).mean()
    df['adjusted_volume'] = df['volume'] / df['volume_ema']
    df['vol_weighted_close_to_open'] = df['close_to_open_return'] * df['adjusted_volume']
    
    # Adaptive Window Calculation based on Volatility
    df['volatility'] = df[['high', 'low', 'close']].rolling(window=20).std().mean(axis=1)
    df['window_size'] = 20 - (df['volatility'] * 5).astype(int)  # Adjust window size based on volatility
    df['window_size'] = df['window_size'].clip(lower=5, upper=20)  # Ensure the window size is between 5 and 20
    
    # Rolling Statistics with Adaptive Window
    def rolling_stats(group, col, window_col):
        return group.rolling(window=group[window_col].iloc[0], min_periods=1).agg([np.mean, np.std])[col]
    
    df['rolling_mean'] = df.groupby('window_size').apply(rolling_stats, 'vol_weighted_close_to_open', 'window_size')['mean']
    df['rolling_std'] = df.groupby('window_size').apply(rolling_stats, 'vol_weighted_close_to_open', 'window_size')['std']
    
    # Final Alpha Factor: Standardized Volume Weighted Close-to-Open Return
    df['alpha_factor'] = (df['vol_weighted_close_to_open'] - df['rolling_mean']) / df['rolling_std']
    
    return df['alpha_factor'].dropna()

# Example usage:
# df = pd.read_csv('your_data.csv', index_col='date', parse_dates=True)
# alpha_factor = heuristics_v2(df)
# print(alpha_factor)
