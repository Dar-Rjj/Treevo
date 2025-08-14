import numpy as np
    
    def log_return(data):
        return np.log(data / data.shift(1))
    
    log_returns = df['close'].apply(log_return).fillna(0)
    step1 = log_returns.rolling(window=5).mean()
    step2 = log_returns.rolling(window=20).std()
    heuristics_matrix = 0.5 * step1 + 0.5 * step2
    
    return heuristics_matrix
