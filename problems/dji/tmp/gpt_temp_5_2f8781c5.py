import pandas as pd
import numpy as np

def heuristics_v2(df):
    def calculate_log_returns(series):
        return np.log(series) - np.log(series.shift(1))
    
    log_returns = df['close'].apply(calculate_log_returns)
    momentum = df['close'] / df['close'].shift(20) - 1
    volume_change = df['volume'] / df['volume'].shift(20) - 1
    
    heuristics_matrix = (log_returns + momentum + volume_change) / 3
    return heuristics_matrix
