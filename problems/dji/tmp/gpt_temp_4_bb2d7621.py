import pandas as pd

def heuristics_v2(df):
    # Calculate simple moving average for closing price over 5 days
    df['SMA_5'] = df['close'].rolling(window=5).mean()
    # Calculate simple moving average for closing price over 10 days
    df['SMA_10'] = df['close'].rolling(window=10).mean()
    # Volume shock indicator: flag if today's volume is more than 2 times the average of the last 5 days
    df['Volume_Shock'] = (df['volume'] > 2 * df['volume'].rolling(window=5).mean()).astype(int)
    # Trend-following metric: difference between SMA_5 and SMA_10
    df['Trend_Following'] = df['SMA_5'] - df['SMA_10']
    # Simple daily return
    df['Daily_Return'] = df['close'].pct_change()
    
    heuristics_matrix = df[['SMA_5', 'SMA_10', 'Volume_Shock', 'Trend_Following', 'Daily_Return']].dropna()
    
    return heuristics_matrix
