import pandas as pd
from scipy.stats import zscore

def heuristics_v2(df):
    log_return = df['close'].apply(lambda x: np.log(x)) - df['close'].shift(1).apply(lambda x: np.log(x))
    volume_zscore = zscore(df['volume'])
    heuristics_matrix = (log_return + volume_zscore).dropna()
    
    return heuristics_matrix
