import pandas as pd
import pandas_ta as ta

def heuristics_v2(df):
    adx = ta.adx(df['high'], df['low'], df['close']).iloc[:, 0]  # ADX is the first column
    cmf = ta.cmf(df['high'], df['low'], df['close'], df['volume'])
    
    heuristics_matrix = (adx * 0.6 + cmf * 0.4).dropna()
    return heuristics_matrix
