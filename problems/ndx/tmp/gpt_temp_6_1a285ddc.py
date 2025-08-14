import pandas as pd

def heuristics_v2(df):
    # Calculate the 10-day and 50-day exponential moving averages of the close price
    ema_10 = df['close'].ewm(span=10, adjust=False).mean()
    ema_50 = df['close'].ewm(span=50, adjust=False).mean()
    
    # Calculate the ratio of the EMAs
    ema_ratio = ema_10 / ema_50
    
    # Calculate the daily return
    df['Return'] = df['close'].pct_change()
    
    # Calculate the 20-day mean absolute deviation of daily returns
    mad_20 = df['Return'].rolling(window=20).apply(lambda x: x.abs().mean(), raw=True)
    
    # Generate the heuristic matrix by dividing the EMA ratio with the mean absolute deviation
    heuristics_matrix = ema_ratio / mad_20
    
    return heuristics_matrix
