import pandas as pd

def heuristics_v2(df):
    # Calculate the 20-day SMA of the closing prices
    sma_20 = df['close'].rolling(window=20).mean()
    
    # Calculate the 30-day standard deviation of the logarithmic returns
    log_returns = df['close'].pct_change().apply(lambda x: np.log(1 + x))
    std_30 = log_returns.rolling(window=30).std()
    
    # Create the heuristic matrix
    heuristics_matrix = (sma_20 / std_30).fillna(0)
    return heuristics_matrix
