import pandas as pd

def heuristics_v2(df):
    log_price_ratio = (df['close'] / df['open']).apply(lambda x: math.log(x))
    roc_volume = df['volume'].pct_change()
    
    heuristics_matrix = 0.7 * log_price_ratio + 0.3 * roc_volume
    return heuristics_matrix
