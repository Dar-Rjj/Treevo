import pandas as pd
    
    # Create a new series for the heuristic factor values
    heuristics_matrix = (df['close'] - df['open']) / df['volume'] + (df['high'] - df['low']) * df['volume']
    
    return heuristics_matrix
