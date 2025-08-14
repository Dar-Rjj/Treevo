import pandas as pd

def heuristics_v2(df):
    price_volume_ratio = df['close'] / df['volume']
    ewma_pv_ratio = price_volume_ratio.ewm(span=20, adjust=False).mean()
    heuristics_matrix = ewma_pv_ratio.apply(lambda x: -1 * (x + 1e-6)).apply(np.log)
    
    return heuristics_matrix
