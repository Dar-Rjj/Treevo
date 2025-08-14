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
    df['true_range'] = df[['high', 'low', 'close']].max(axis=1) - df[['high', 'low', 'close']].min(axis=1)
    initial_window = 20
    df['volatility'] = df['true_range'].rolling(window=initial_window).std()

    # Adaptive Window Calculation
    volatility_threshold = df['volatility'].mean()
    df['window_size'] = np.where(df['volatility'] > volatility_threshold, 10, 30)

    # Rolling Statistics with Adaptive Window
    df['rolling_mean'] = df.apply(lambda row: df.loc[:row.name]['volume_weighted_return'].rolling(window=row['window_size']).mean().iloc[-1], axis=1)
    df['rolling_std'] = df.apply(lambda row: df.loc[:row.name]['volume_weighted_return'].rolling(window=row['window_size']).std().iloc[-1], axis=1)

    # Final Factor: Standardized Rolling Mean
    df['factor'] = (df['rolling_mean'] - df['rolling_mean'].mean()) / df['rolling_mean'].std()

    return df['factor'].dropna()
