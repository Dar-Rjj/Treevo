import pandas as pd
    df['avg_vol_5d'] = df['volume'].rolling(window=5).mean()
    heuristics_factor = (df['avg_vol_5d'] / df['close']) - df['close'].rolling(window=30).mean()
    heuristics_matrix = heuristics_factor.dropna()
    return heuristics_matrix
