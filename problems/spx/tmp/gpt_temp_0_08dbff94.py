import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df, n=20):
    # Calculate Daily Price Acceleration
    df['Close_diff'] = df['close'].diff()
    df['Close_acceleration'] = df['Close_diff'].diff()
    df['Smoothed_acceleration'] = df['Close_acceleration'].rolling(window=n).sum()

    # Incorporate Volume Adjusted Inertia
    positive_volume = (df['close'] > df['close'].shift(1)) * (df['close'] - df['close'].shift(1)) * df['volume']
    negative_volume = (df['close'] <= df['close'].shift(1)) * -(df['close'].shift(1) - df['close']) * df['volume']
    
    df['Positive_volume_sum'] = positive_volume.rolling(window=n).sum()
    df['Negative_volume_sum'] = negative_volume.rolling(window=n).sum()
    
    df['Intermediate_alpha'] = (df['Smoothed_acceleration'] * df['Positive_volume_sum']) / df['Negative_volume_sum'].abs()

    # Integrate Enhanced Price-Volume and Spread Dynamics
    df['High_low_spread'] = df['high'] - df['low']
    df['Open_close_spread'] = df['open'] - df['close']

    df['Pearson_corr'] = df['close'].rolling(window=n).corr(df['volume'])
    df['Kendall_tau'] = df['High_low_spread'].rolling(window=n).apply(lambda x: pd.Series(x).kendalltau(df['volume'])[0])
    
    df['Avg_high_low_spread'] = df['High_low_spread'].rolling(window=n).mean()
    df['Relative_high_low_spread'] = df['High_low_spread'] / df['Avg_high_low_spread']
    
    df['Avg_open_close_spread'] = df['Open_close_spread'].rolling(window=n).mean()
    df['Relative_open_close_spread'] = df['Open_close_spread'] / df['Avg_open_close_spread']

    # Final Alpha Factor
    df['Final_alpha'] = df['Intermediate_alpha'] * df['Pearson_corr'] * df['Kendall_tau'] * df['Relative_high_low_spread'] + df['Relative_open_close_spread']

    return df['Final_alpha']
