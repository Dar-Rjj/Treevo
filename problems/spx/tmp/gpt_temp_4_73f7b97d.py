import pandas as pd
import pandas as pd

def heuristics_v2(df, n=14):
    # Calculate Daily Price Momentum
    df['close_diff'] = df['close'] - df['close'].shift(n)
    df['smoothed_momentum'] = df['close_diff'].rolling(window=n, min_periods=n).sum()

    # Incorporate Volume Adjusted Inertia
    df['volume_flow'] = df.apply(lambda row: row['volume'] if row['close'] > row['close'].shift(1) else -row['volume'], axis=1)
    
    df['positive_volume_sum'] = df[df['volume_flow'] > 0]['volume_flow'].rolling(window=n, min_periods=n).sum()
    df['negative_volume_sum'] = df[df['volume_flow'] < 0]['volume_flow'].rolling(window=n, min_periods=n).sum().abs()
    
    df['intermediate_alpha'] = (df['smoothed_momentum'] * df['positive_volume_sum']) / df['negative_volume_sum']

    # Integrate Price-Volume Correlation
    df['price_vol_corr'] = df['close'].rolling(window=n, min_periods=n).corr(df['volume'])
    
    df['final_alpha'] = df['intermediate_alpha'] * df['price_vol_corr']
    
    return df['final_alpha'].dropna()
