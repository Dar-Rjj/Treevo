import pandas as pd

def heuristics_v2(df):
    df['avg_high_low'] = (df['high'] + df['low']) / 2
    df['close_to_avg_ratio'] = df['close'] / df['avg_high_low']
    heuristics_matrix = df['close_to_avg_ratio'].rolling(window=5).apply(lambda x: (x * df.loc[x.index, 'volume']).sum() / df.loc[x.index, 'volume'].sum(), raw=False)
    return heuristics_matrix
