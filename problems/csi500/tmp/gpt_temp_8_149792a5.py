import numpy as np
def heuristics_v2(df):
    # Calculate Price Momentum
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
