import pandas as pd

def heuristics_v2(df):
    # Calculate log returns
    log_returns = df['close'].apply(lambda x: math.log(x)).diff()
    
    # Calculate EMA of the closing prices
    ema_close = df['close'].ewm(span=10, adjust=False).mean()
    
    # Calculate the standard deviation of log returns to represent volatility
    volatility = log_returns.rolling(window=10).std()
    
    # Composite heuristic
    heuristics_matrix = ema_close + volatility
    
    return heuristics_matrix
