import pandas as pd

def heuristics_v2(df):
    # Calculate the 20-day and 100-day weighted moving averages of the close price
    wma_20 = df['close'].rolling(window=20).apply(lambda x: (x * range(1, len(x) + 1)).sum() / sum(range(1, len(x) + 1)), raw=False)
    wma_100 = df['close'].rolling(window=100).apply(lambda x: (x * range(1, len(x) + 1)).sum() / sum(range(1, len(x) + 1)), raw=False)
    
    # Calculate the ratio between the WMAs
    wma_ratio = wma_20 / wma_100
    
    # Calculate the daily log return
    df['Log_Return'] = np.log(df['close']).diff()
    
    # Calculate the 30-day standard deviation of daily log returns
    std_30 = df['Log_Return'].rolling(window=30).std()
    
    # Generate the heuristic matrix by multiplying the WMA ratio with the STD
    heuristics_matrix = wma_ratio * std_30
    
    return heuristics_matrix
