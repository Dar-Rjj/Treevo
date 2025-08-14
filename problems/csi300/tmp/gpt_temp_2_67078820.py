import pandas as pd

def heuristics_v2(df):
    # Calculate daily return
    df['return'] = df['close'].pct_change()
    
    # Heuristic 1: Moving Average Crossover
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['sma_200'] = df['close'].rolling(window=200).mean()
    df['h1'] = df['sma_50'] - df['sma_200']
    
    # Heuristic 2: Price-Volume Trend
    df['pvt'] = (df['volume'] * (df['close'] - df['close'].shift(1))).cumsum()
    df['h2'] = df['pvt'].pct_change()
    
    # Heuristic 3: Relative Strength Index
    delta = df['close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['h3'] = df['rsi'].pct_change()
    
    # Creating the heuristics matrix
    heuristics_matrix = df[['h1', 'h2', 'h3']].dropna()
    
    return heuristics_matrix
