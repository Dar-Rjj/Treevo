import pandas as pd

def heuristics_v2(df):
    # Calculate the log difference between close and open prices
    log_diff = (df['close'] / df['open']).apply(lambda x: math.log(x))
    
    # Smooth the volume using a 10-day exponential moving average
    ema_volume = df['volume'].ewm(span=10, adjust=False).mean()
    
    # Calculate the volatility as the standard deviation of (high - low) over a 20-day window
    volatility = (df['high'] - df['low']).rolling(window=20).std()
    
    # Generate the heuristic factor
    heuristics_matrix = log_diff + ema_volume + volatility
    return heuristics_matrix
