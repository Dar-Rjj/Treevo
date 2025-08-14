import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

def heuristics_v2(df, n=20):
    # Calculate Daily Price Acceleration
    df['Close_diff_1'] = df['close'].diff()
    df['Close_diff_2'] = df['close'].diff(2)
    df['Second_difference'] = df['Close_diff_1'] - df['Close_diff_1'].shift(1)
    df['Smoothed_acceleration'] = df['Second_difference'].rolling(window=n).sum()

    # Incorporate Volume Adjusted Inertia
    df['Positive_volume'] = 0
    df['Negative_volume'] = 0
    for i in range(1, n+1):
        df.loc[df['close'] > df['close'].shift(i), 'Positive_volume'] += df['volume'] * (df['close'] - df['close'].shift(i))
        df.loc[df['close'] <= df['close'].shift(i), 'Negative_volume'] -= df['volume'] * (df['close'].shift(i) - df['close'])
    
    df['Sum_positive_volume'] = df['Positive_volume'].rolling(window=n).sum()
    df['Sum_negative_volume'] = df['Negative_volume'].rolling(window=n).sum()
    df['Intermediate_alpha_factor'] = df['Smoothed_acceleration'] * df['Sum_positive_volume'] / df['Sum_negative_volume'].abs()

    # Integrate Enhanced Price-Volume Dynamics
    df['High_low_spread'] = df['high'] - df['low']
    
    def rolling_pearsonr(x, y, window):
        return [pearsonr(x.iloc[i:i+window], y.iloc[i:i+window])[0] for i in range(len(x) - window + 1)]
    
    def rolling_spearmanr(x, y, window):
        return [spearmanr(x.iloc[i:i+window], y.iloc[i:i+window])[0] for i in range(len(x) - window + 1)]
    
    df['Pearson_correlation'] = np.nan
    df['Spearman_correlation'] = np.nan
    
    if len(df) >= n:
        df['Pearson_correlation'].iloc[n-1:] = rolling_pearsonr(df['close'], df['volume'], n)
        df['Spearman_correlation'].iloc[n-1:] = rolling_spearmanr(df['High_low_spread'], df['volume'], n)
    
    df['Final_alpha_factor'] = df['Intermediate_alpha_factor'] * df['Pearson_correlation'] * df['Spearman_correlation']
    
    return df['Final_alpha_factor']

# Example usage:
# df = pd.read_csv('your_data.csv', parse_dates=['date'], index_col='date')
# alpha_factor = heuristics_v2(df)
# print(alpha_factor)
