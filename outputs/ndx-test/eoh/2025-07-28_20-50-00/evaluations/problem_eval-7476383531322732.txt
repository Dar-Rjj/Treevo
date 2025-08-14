import pandas as pd

def heuristics_v2(df):
    # Calculate log returns
    df['log_returns'] = (df['close']).apply(np.log).diff()
    
    # Compute the 20-day standard deviation of log returns as a measure of volatility
    df['volatility'] = df['log_returns'].rolling(window=20).std()
    
    # Momentum calculation using 10-day simple moving average of close prices
    momentum = df['close'].rolling(window=10).mean().diff()
    
    # Generate the heuristic matrix combining momentum and volatility
    heuristics_matrix = momentum - df['volatility']
    
    return heuristics_matrix
