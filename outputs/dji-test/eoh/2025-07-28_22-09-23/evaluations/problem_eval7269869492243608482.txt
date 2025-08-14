import pandas as pd

def heuristics_v2(df):
    # Calculate the momentum factor (close price change over 10 days)
    momentum = df['close'].pct_change(periods=10)
    
    # Calculate the volatility factor (standard deviation of daily returns over 30 days)
    volatility = df['close'].pct_change().rolling(window=30).std()
    
    # Calculate the liquidity factor (average volume over 30 days)
    liquidity = df['volume'].rolling(window=30).mean()
    
    # Combine the factors into a single heuristic
    heuristics_matrix = (momentum * 0.4 + volatility * 0.3 + (1/liquidity) * 0.3).apply(lambda x: pd.np.log(1 + x))
    
    return heuristics_matrix
