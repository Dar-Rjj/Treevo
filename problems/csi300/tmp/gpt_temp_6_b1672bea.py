import pandas as pd

def heuristics_v2(df):
    # Weighted Moving Averages
    wma_5 = df['close'].rolling(window=5).apply(lambda prices: (prices * 5).sum() / 15)
    wma_10 = df['close'].rolling(window=10).apply(lambda prices: (prices * 10).sum() / 55)
    wma_ratio = (wma_5 / wma_10) - 1
    
    # Customized Volatility Measure
    daily_returns = df['close'].pct_change()
    volatility = daily_returns.rolling(window=10).std()
    
    # Volume Rate of Change (VROC) with 10-day window
    vroc = df['volume'].pct_change(periods=10)
    
    # Composite heuristic
    heuristics_matrix = (wma_ratio + volatility + vroc) / 3
    return heuristics_matrix
