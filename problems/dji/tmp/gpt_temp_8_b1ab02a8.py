import pandas as pd

def heuristics_v2(df):
    # Calculate 20-day EMA of the close price
    ema_20 = df['close'].ewm(span=20, adjust=False).mean()
    
    # Calculate 14-day standard deviation (volatility) of the close price
    std_14 = df['close'].rolling(window=14).std()
    
    # Logarithm of the trading volume
    log_volume = df['volume'].apply(lambda x: max(1, x)).apply(lambda x: math.log(x))
    
    # Create the heuristic matrix
    heuristics_matrix = (ema_20 / std_14) * log_volume
    return heuristics_matrix
