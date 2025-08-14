import pandas as pd
import numpy as np

def heuristics_v2(df):
    log_changes = np.log(df[['Open', 'High', 'Low', 'Close']]) - np.log(df[['Open', 'High', 'Low', 'Close']].shift(1))
    volume_sqrt = np.sqrt(df['Volume'])
    weights = np.array([0.4, 0.3, 0.2, 0.1])
    heuristics_matrix = (log_changes * weights).sum(axis=1) + volume_sqrt
    return heuristics_matrix
