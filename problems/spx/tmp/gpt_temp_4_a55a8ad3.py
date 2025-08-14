import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

def heuristics_v2(df, n=20):
    # Calculate Daily Price Jerk
    df['close_t'] = df['close']
    df['close_t-1'] = df['close'].shift(1)
    df['close_t-n'] = df['close'].shift(n)
    
    df['jerk'] = df['close_t'] - 3 * df['close_t-1'] + 3 * df['close_t-n'] - df['close_t'].shift(2*n)
    df['smoothed_jerk'] = df['jerk'].rolling(window=n).sum()
    
    # Incorporate Volume Adjusted Inertia
    df['positive_volume'] = df.apply(lambda row: row['volume'] * (row['close'] - row['close'].shift(1)) if row['close'] > row['close'].shift(1) else 0, axis=1)
    df['negative_volume'] = df.apply(lambda row: -row['volume'] * (row['close'].shift(1) - row['close']) if row['close'] <= row['close'].shift(1) else 0, axis=1)
    
    df['sum_positive_volume'] = df['positive_volume'].rolling(window=n).sum()
    df['sum_negative_volume'] = df['negative_volume'].rolling(window=n).sum().abs()
    
    df['intermediate_alpha'] = (df['smoothed_jerk'] * df['sum_positive_volume']) / df['sum_negative_volume']
    
    # Integrate Enhanced Price-Volume and Spread Dynamics
    df['high_low_spread'] = df['high'] - df['low']
    df['pearson_corr'] = df['close'].rolling(window=n).corr(df['volume'])
    df['spearman_corr'] = df.rolling(window=n).apply(lambda x: spearmanr(x['high_low_spread'], x['volume'])[0], raw=False)
    
    df['avg_high_low_spread'] = df['high_low_spread'].rolling(window=n).mean()
    df['relative_high_low_spread'] = df['high_low_spread'] / df['avg_high_low_spread']
    
    df['final_alpha'] = df['intermediate_alpha'] * df['pearson_corr'] * df['spearman_corr'] * df['relative_high_low_spread']
    
    return df['final_alpha']
