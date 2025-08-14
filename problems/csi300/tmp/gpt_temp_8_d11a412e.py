import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Close-to-Open Return
    df['close_to_open_return'] = df['open'].shift(-1) - df['close']
    
    # Volume Weighting
    df['volume_weighted_return'] = df['close_to_open_return'] * df['volume']
    
    # Dynamic Volatility Calculation
    def calculate_volatility(df, window=5):
        high_low_diff = df['high'] - df['low']
        high_close_diff = np.abs(df['high'] - df['close'].shift(1))
        low_close_diff = np.abs(df['low'] - df['close'].shift(1))
        tr = np.maximum(high_low_diff, high_close_diff, low_close_diff)
        return tr.rolling(window).std()
    
    df['volatility'] = calculate_volatility(df)
    
    # Adjust Window Size
    df['window_size'] = np.where(df['volatility'] > df['volatility'].mean(), 10, 30)
    
    # Liquidity Factor
    df['avg_volume_20'] = df['volume'].rolling(20).mean()
    df['liquidity_adjusted_return'] = df['volume_weighted_return'] * df['avg_volume_20']
    
    # Rolling Statistics with Adaptive Window
    def rolling_stats(series, window_series):
        mean = series.rolling(window_series).mean()
        std = series.rolling(window_series).std()
        return mean, std
    
    df['adaptive_mean'], df['adaptive_std'] = rolling_stats(df['liquidity_adjusted_return'], df['window_size'])
    
    # Alpha factor
    df['alpha_factor'] = df['adaptive_mean'] / df['adaptive_std']
    
    return df['alpha_factor'].dropna()

# Example usage
# df = pd.read_csv('market_data.csv', index_col='date', parse_dates=True)
# alpha_factor = heuristics_v2(df)
