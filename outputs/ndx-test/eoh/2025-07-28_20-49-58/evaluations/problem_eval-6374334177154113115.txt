import pandas as pd

    def calculate_returns(column, n):
        return column.pct_change(n)

    # Calculate 30-day returns
    returns = calculate_returns(df['close'], 30)
    
    # Calculate 10-day exponentially weighted moving average of the closing price
    ewma_10 = df['close'].ewm(span=10, adjust=False).mean()
    
    # Calculate modified price-to-volume ratio
    pv_ratio = (df['close'] / df['volume']).rolling(window=30).mean()
    
    # Calculate 30-day standard deviation of returns
    std_30 = returns.rolling(window=30).std()
    
    # Adjust the heuristics matrix with volatility
    heuristics_matrix = pv_ratio + (ewma_10 - df['close']) + std_30
    
    return heuristics_matrix
