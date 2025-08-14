import pandas as pd

def heuristics_v2(df):
    # Calculate simple moving averages
    ma_50 = df['close'].rolling(window=50).mean()
    ma_200 = df['close'].rolling(window=200).mean()
    
    # Momentum: difference in closing price over the last 10 days
    momentum_10 = df['close'] - df['close'].shift(10)
    
    # Volatility: standard deviation of daily returns over the last 30 days
    daily_returns = (df['close'] / df['close'].shift(1)) - 1
    volatility_30 = daily_returns.rolling(window=30).std()
    
    # Combine factors into a DataFrame
    heuristics_matrix = pd.DataFrame({
        'MA_50': ma_50,
        'MA_200': ma_200,
        'Momentum_10': momentum_10,
        'Volatility_30': volatility_30
    }).dropna()
    
    # Return the heuristics matrix
    return heuristics_matrix
