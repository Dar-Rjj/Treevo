import pandas as pd
import numpy as pd

def heuristics_v2(df):
    high_low_diff = df['high'] - df['low']
    prev_close = df['close'].shift(1)
    price_ratio = high_low_diff / prev_close
    max_volume_30 = df['volume'].rolling(window=30).max()
    vol_log_ratio = np.log(max_volume_30 / df['volume'])
    heuristics_matrix = price_ratio + vol_log_ratio
    return heuristics_matrix
