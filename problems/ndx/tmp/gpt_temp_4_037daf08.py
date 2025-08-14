import pandas as pd

def heuristics_v2(df):
    # Calculate the 5-day and 20-day average volume
    avg_volume_5 = df['volume'].rolling(window=5).mean()
    avg_volume_20 = df['volume'].rolling(window=20).mean()
    
    # Calculate the volume ratio
    volume_ratio = avg_volume_5 / avg_volume_20
    
    # Calculate the 10-day WMA of the closing price
    weights = pd.Series(range(1, 11))
    wma_10 = df['close'].rolling(window=10).apply(lambda prices: (prices * weights).sum() / weights.sum(), raw=False)
    
    # Calculate the 30-day standard deviation of daily returns
    daily_returns = df['close'].pct_change()
    std_30 = daily_returns.rolling(window=30).std()
    
    # Combine factors into the heuristics matrix
    heuristics_matrix = volume_ratio + (wma_10 - df['close']) + std_30
    
    return heuristics_matrix
