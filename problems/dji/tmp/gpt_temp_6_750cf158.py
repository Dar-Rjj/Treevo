import pandas as pd

def heuristics_v2(df):
    # Momentum factor: 10-day return
    momentum = df['close'].pct_change(10)
    
    # Volatility factor: standard deviation of daily returns over the last 30 days
    daily_returns = df['close'].pct_change()
    volatility = daily_returns.rolling(window=30).std()
    
    # Volume growth: ratio of today's volume to the average volume over the past 5 days
    avg_volume_5d = df['volume'].rolling(window=5).mean()
    volume_growth = df['volume'] / avg_volume_5d
    
    # Combine factors into a DataFrame
    heuristics_matrix = pd.concat([momentum, volatility, volume_growth], axis=1)
    heuristics_matrix.columns = ['Momentum', 'Volatility', 'Volume_Growth']
    
    return heuristics_matrix
