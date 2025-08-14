import numpy as np
def heuristics_v2(df):
    # Calculate Daily Log Return
    df['log_return'] = np.log(df['close']) - np.log(df['close'].shift(1))
