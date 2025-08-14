import pandas as pd
import numpy as pd

def heuristics_v2(df):
    # On-Balance Volume (OBV)
    obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    obv_ema = obv.ewm(span=20, adjust=False).mean()
    
    # Logarithmic Difference between Close and Open Prices
    close_open_log_diff = np.log(df['close'] / df['open'])
    
    # Composite heuristic matrix
    heuristics_matrix = 0.6 * obv_ema + 0.4 * close_open_log_diff
    
    return heuristics_matrix
