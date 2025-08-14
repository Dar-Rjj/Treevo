import numpy as np

def heuristics_v2(df):
    # Define the lookback window for calculations
    window_size = 20
    
    # Function to compute DFA
    def dfa(series, n=window_size):
        X = np.cumsum(series - np.mean(series))
        Y = [X[i:i + n] for i in range(0, len(X) - n)]
        f = np.sqrt(np.array([np.var(y) for y in Y]))
        return np.mean(f)
    
    # Compute the DFA of the closing prices over a rolling window
    df['dfa_close'] = df['close'].rolling(window=window_size).apply(dfa, raw=False)
    
    # Compute the cumulative sum of the trading volume
    df['cum_volume'] = df['volume'].cumsum()
    
    # Construct the heuristics matrix by combining DFA and cumulative volume
    heuristics_matrix = df['dfa_close'] * df['cum_volume']
    
    return heuristics_matrix
