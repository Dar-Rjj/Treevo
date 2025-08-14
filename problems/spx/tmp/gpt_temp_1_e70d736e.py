import pandas as pd
import pandas as pd

def heuristics_v2(df, n=10):
    # Calculate Daily Price Momentum
    df['close_diff'] = df['close'].diff()
    df['smoothed_momentum'] = df['close_diff'].rolling(window=n).sum()

    # Compute Cumulative Volume Flow
    df['volume_flow'] = df.apply(lambda row: row['volume'] if row['close'] > df.loc[row.name - pd.Timedelta(days=1), 'close'] else -row['volume'], axis=1)
    
    # Sum positive and negative volumes separately for the past n days
    df['positive_volume_flow'] = df['volume_flow'].apply(lambda x: max(x, 0)).rolling(window=n).sum()
    df['negative_volume_flow'] = df['volume_flow'].apply(lambda x: min(x, 0)).rolling(window=n).sum().abs()

    # Combine Momentum with Cumulative Volume Flow
    df['alpha_factor'] = (df['smoothed_momentum'] * df['positive_volume_flow']) / df['negative_volume_flow']

    return df['alpha_factor']
