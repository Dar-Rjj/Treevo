import numpy as np
def heuristics_v2(df):
    # Calculate Logarithmic Daily Returns
    df['log_returns'] = np.log(df['close']) - np.log(df['close'].shift(1))
