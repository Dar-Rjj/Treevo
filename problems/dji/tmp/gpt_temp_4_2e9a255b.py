import pandas as pd

def heuristics_v2(df):
    def calculate_momentum(row):
        return (row['close'] - row['open']) / row['open']
    
    def calculate_smoothed_volatility(row):
        return (row['high'] - row['low']).rolling(window=5).mean()
    
    def calculate_volume_ratio(row):
        return row['volume'] / df['volume'].rolling(window=30).mean().shift(1)[row.name]

    df['momentum'] = df.apply(calculate_momentum, axis=1)
    df['smoothed_volatility'] = df.apply(calculate_smoothed_volatility, axis=1)
    df['volume_ratio'] = df.apply(calculate_volume_ratio, axis=1)
    
    heuristics_matrix = 0.5 * df['momentum'] + 0.2 * df['smoothed_volatility'] + 0.3 * df['volume_ratio']
    
    return heuristics_matrix
