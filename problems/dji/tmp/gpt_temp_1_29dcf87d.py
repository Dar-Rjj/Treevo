import numpy as np
    
    df['log_returns'] = np.log(df['close']) - np.log(df['close'].shift(1))
    heuristics_matrix = df['log_returns'].rolling(window=20).std()
    return heuristics_matrix
