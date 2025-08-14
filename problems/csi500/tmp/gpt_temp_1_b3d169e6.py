import numpy as np
def heuristics_v2(df):
    # Daily Log Returns
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
