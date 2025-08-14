import pandas as pd

def heuristics_v2(df):
    def calculate_roc(column, n):
        return (column - column.shift(n)) / column.shift(n)

    # Calculate ROC over a 10-day period
    roc = calculate_roc(df['close'], 10)
    
    # Calculate the 20-day standard deviation of daily log returns for volatility
    log_returns = df['close'].apply(lambda x: np.log(x))
    volatility = log_returns.rolling(window=20).std()
    
    # Calculate 50-day simple moving average of the closing price
    sma50 = df['close'].rolling(window=50).mean()
    
    # Distance between the current close and the 50-day SMA
    distance_sma50 = df['close'] - sma50
    
    # Combine factors
    heuristics_matrix = roc + volatility + distance_sma50
    
    return heuristics_matrix
