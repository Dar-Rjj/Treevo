import numpy as np
def heuristics_v2(df):
    # Calculate Daily Return
    df['Daily_Return'] = df['close'] - df['close'].shift(1)
    
    # Calculate Price Gap
    df['Price_Gap'] = df['open'] - df['close'].shift(1)
    
    # Adjust Daily Return by Price Gap with Exponential Decay
    def exponential_decay(x, half_life=1):
        return np.exp(-np.log(2) / half_life * x)
