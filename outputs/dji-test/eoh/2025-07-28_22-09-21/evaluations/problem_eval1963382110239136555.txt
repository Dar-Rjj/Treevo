import pandas as pd

def heuristics_v2(df):
    # Trend strength - based on moving averages
    df['SMA_5'] = df['close'].rolling(window=5).mean()
    df['SMA_10'] = df['close'].rolling(window=10).mean()
    df['Trend_Strength'] = df['SMA_5'] - df['SMA_10']
    
    # Volatility - standard deviation of daily returns over the last 10 days
    df['Daily_Return'] = df['close'].pct_change()
    df['Volatility'] = df['Daily_Return'].rolling(window=10).std()
    
    # Volume Anomaly - flagging days with unusually high or low volume
    df['Volume_Mean'] = df['volume'].rolling(window=30).mean()
    df['Volume_Ratio'] = df['volume'] / df['Volume_Mean']
    
    # Combining factors into a single heuristics matrix
    heuristics_matrix = df[['Trend_Strength', 'Volatility', 'Volume_Ratio']].copy()
    
    return heuristics_matrix
