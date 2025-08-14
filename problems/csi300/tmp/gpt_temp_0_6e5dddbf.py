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
    df['high_low_close_mean'] = (df['high'] + df['low'] + df['close']) / 3
    initial_window = 20
    df['volatility'] = df['high_low_close_mean'].rolling(window=initial_window).std()
    
    # Adjust Window Size Based on Volatility
    volatility_threshold = df['volatility'].quantile(0.5)
    df['window_size'] = np.where(df['volatility'] > volatility_threshold, 
                                 initial_window // 2, 
                                 initial_window * 2)
    
    # Calculate Rolling Statistics with Adaptive Window
    def rolling_stats(x, window):
        return (x.rolling(window=window).mean(), x.rolling(window=window).std())
    
    df['rolling_mean'] = 0
    df['rolling_std'] = 0
    for i in range(len(df)):
        window = int(df.loc[df.index[i], 'window_size'])
        if i >= window:
            mean, std = rolling_stats(df['volume_weighted_return'], window)
            df.loc[df.index[i], 'rolling_mean'] = mean.iloc[i]
            df.loc[df.index[i], 'rolling_std'] = std.iloc[i]
    
    # Final Alpha Factor: Standardized Rolling Mean
    df['alpha_factor'] = df['rolling_mean'] / (df['rolling_std'] + 1e-6)
    
    return df['alpha_factor'].dropna()

# Example usage:
# df = pd.read_csv('your_data.csv', index_col='date', parse_dates=True)
# alpha_factor = heuristics_v2(df)
# print(alpha_factor)
