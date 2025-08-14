import pandas as pd

def heuristics_v2(df):
    # Calculate the 10-day and 50-day weighted moving averages of the close price
    weights_10 = pd.Series(range(1, 11))
    weights_50 = pd.Series(range(1, 51))
    wma_10 = df['close'].rolling(window=10).apply(lambda x: (x * weights_10).sum() / weights_10.sum(), raw=False)
    wma_50 = df['close'].rolling(window=50).apply(lambda x: (x * weights_50).sum() / weights_50.sum(), raw=False)
    
    # Calculate the ratio between the WMAs
    wma_ratio = wma_10 / wma_50
    
    # Calculate the daily return
    df['Return'] = df['close'].pct_change()
    
    # Calculate the 20-day mean and standard deviation of daily returns
    mean_20 = df['Return'].rolling(window=20).mean()
    std_dev_20 = df['Return'].rolling(window=20).std()
    
    # Calculate the 20-day coefficient of variation of daily returns
    cv_20 = std_dev_20 / mean_20
    
    # Generate the heuristic matrix by multiplying the WMA ratio with the CV
    heuristics_matrix = wma_ratio * cv_20
    
    return heuristics_matrix
