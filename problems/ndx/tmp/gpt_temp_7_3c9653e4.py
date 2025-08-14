import pandas as pd

def heuristics_v2(df):
    # Calculate the 20-day and 60-day weighted moving averages of the close price
    wma_20 = df['close'].rolling(window=20).apply(lambda x: (x * pd.Series(range(1, 21))).sum() / 210, raw=False)
    wma_60 = df['close'].rolling(window=60).apply(lambda x: (x * pd.Series(range(1, 61))).sum() / 1830, raw=False)
    
    # Calculate the ratio between the WMAs
    wma_ratio = wma_20 / wma_60
    
    # Calculate the daily return
    df['Return'] = df['close'].pct_change()
    
    # Calculate the 30-day coefficient of variation of daily returns
    cv_30 = df['Return'].rolling(window=30).std() / df['Return'].rolling(window=30).mean()
    
    # Generate the heuristic matrix by multiplying the WMA ratio with the CV
    heuristics_matrix = wma_ratio * cv_30
    
    return heuristics_matrix
