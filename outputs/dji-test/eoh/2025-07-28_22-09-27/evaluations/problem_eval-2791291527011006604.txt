import pandas as pd

def heuristics_v2(df):
    # Calculate the logarithmic difference between high and low prices
    log_diff = (df['high'] / df['low']).apply(lambda x: math.log(x))
    
    # Calculate the 7-day exponential moving average of the closing prices
    ema = df['close'].ewm(span=7, adjust=False).mean()
    
    # Compute the heuristics factor
    heuristics_matrix = log_diff + (df['close'] - ema)
    
    return heuristics_matrix
