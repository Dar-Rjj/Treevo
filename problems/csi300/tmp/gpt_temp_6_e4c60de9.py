import pandas as pd

def heuristics_v2(df):
    high_low_range = df['high'] - df['low']
    range_autocorr = high_low_range.rolling(window=10).apply(lambda x: x.autocorr(lag=1), raw=False)
    volume_ratio = df['volume'] / df['volume'].rolling(window=10).mean()
    
    raw_factor = range_autocorr * volume_ratio
    heuristics_matrix = raw_factor.rename('heuristics_v2')
    
    return heuristics_matrix
