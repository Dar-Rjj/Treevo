import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Close-to-Open Return
    df['close_to_open_return'] = df['open'].shift(-1) - df['close']
    
    # Volume Weighting
    df['volume_weighted_return'] = df['close_to_open_return'] * df['volume']
    
    # Adaptive Window Calculation
    fixed_window = 30  # Fixed window for initial volatility calculation
    df['volatility'] = df[['high', 'low', 'close']].rolling(window=fixed_window).std().mean(axis=1)
    
    def adjust_window(volatility, threshold_high=0.5, threshold_low=0.2, min_window=10, max_window=60):
        if volatility > threshold_high:
            return max(min_window, int(max_window * (1 - (volatility - threshold_high))))
        elif volatility < threshold_low:
            return min(max_window, int(min_window / (1 - (threshold_low - volatility))))
        else:
            return fixed_window
    
    df['window_size'] = df['volatility'].apply(adjust_window)
    
    # Rolling Statistics with Adaptive Window
    def rolling_stats(group):
        mean = group.rolling(window=group.name).mean()
        std = group.rolling(window=group.name).std()
        return pd.DataFrame({'mean': mean, 'std': std})
    
    df['group'] = df.groupby('window_size').cumcount() + 1
    df.set_index(['group'], append=True, inplace=True)
    result = df.groupby(['window_size', 'group'])['volume_weighted_return'].apply(rolling_stats).reset_index()
    result.set_index('date', inplace=True)
    
    # Final factor value: Z-score of the rolling mean
    result['factor_value'] = (result['mean'] - result['mean'].mean()) / result['mean'].std()
    
    return result['factor_value']

# Example usage:
# df = pd.read_csv('your_data.csv', parse_dates=['date'], index_col='date')
# factor_values = heuristics_v2(df)
