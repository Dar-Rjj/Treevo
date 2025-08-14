import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

def heuristics_v2(df, n=10):
    # Calculate Daily Price Jerk
    df['close_t'] = df['close']
    df['close_t-1'] = df['close'].shift(1)
    df['close_t-n'] = df['close'].shift(n)
    
    df['jerk'] = (df['close_t'] - 3 * df['close_t-1'] + 3 * df['close_t-n'] - df['close_t-n'].shift(n))
    df['smoothed_jerk'] = df['jerk'].rolling(window=n).sum()
    
    # Incorporate Volume Adjusted Inertia
    def dynamic_volume_flow(row, n):
        if row['close'] > row['close'].shift(1):
            return row['volume'] * (row['close'] - row['close'].shift(1))
        else:
            return -row['volume'] * (row['close'].shift(1) - row['close'])
    
    df['dynamic_volume_flow'] = df.apply(dynamic_volume_flow, axis=1, args=(n,))
    df['positive_volume'] = df['dynamic_volume_flow'].apply(lambda x: x if x > 0 else 0)
    df['negative_volume'] = df['dynamic_volume_flow'].apply(lambda x: x if x < 0 else 0)
    
    df['sum_positive_volume'] = df['positive_volume'].rolling(window=n).sum()
    df['sum_negative_volume'] = df['negative_volume'].rolling(window=n).sum().abs()
    
    df['intermediate_alpha_factor'] = df['smoothed_jerk'] * df['sum_positive_volume'] / df['sum_negative_volume']
    
    # Integrate Enhanced Price-Volume and Spread Dynamics
    df['high_low_spread'] = df['high'] - df['low']
    df['open_close_spread'] = df['open'] - df['close']
    
    def rolling_pearson_corr(series1, series2, window):
        return series1.rolling(window=window).corr(series2)
    
    def rolling_spearman_corr(series1, series2, window):
        return series1.rolling(window=window).apply(lambda x: spearmanr(x, series2[x.index][::-1])[0], raw=False)
    
    df['pearson_corr'] = rolling_pearson_corr(df['close'], df['volume'], n)
    df['spearman_corr'] = rolling_spearman_corr(df['high_low_spread'], df['volume'], n)
    
    df['avg_high_low_spread'] = df['high_low_spread'].rolling(window=n).mean()
    df['relative_high_low_spread'] = df['high_low_spread'] / df['avg_high_low_spread']
    
    df['avg_open_close_spread'] = df['open_close_spread'].rolling(window=n).mean()
    df['relative_open_close_spread'] = df['open_close_spread'] / df['avg_open_close_spread']
    
    df['alpha_factor'] = (df['intermediate_alpha_factor'] * df['pearson_corr'] * df['spearman_corr'] 
                          * df['relative_high_low_spread'] * df['relative_open_close_spread'])
    
    return df['alpha_factor']
