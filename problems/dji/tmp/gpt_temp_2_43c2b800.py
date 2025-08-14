import pandas as pd

def heuristics_v2(df):
    def calculate_log_return(row):
        return np.log(row['close']) - np.log(row['open'])
    
    def calculate_range_ratio(row):
        return (row['high'] - row['low']) / row['close']
    
    def calculate_volume_surge(row):
        return (row['volume'] - df['volume'].rolling(window=5).mean().shift(1)[row.name]) / df['volume'].rolling(window=5).mean().shift(1)[row.name]

    df['log_return'] = df.apply(calculate_log_return, axis=1)
    df['range_ratio'] = df.apply(calculate_range_ratio, axis=1)
    df['volume_surge'] = df.apply(calculate_volume_surge, axis=1)
    
    heuristics_matrix = 0.3 * df['log_return'] + 0.4 * df['range_ratio'] + 0.3 * df['volume_surge']
    
    return heuristics_matrix
