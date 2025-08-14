import pandas as pd

def heuristics_v2(df):
    # Calculate short-term (10 days) and long-term (50 days) exponential moving averages for 'close' price
    df['EMA_10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    # Compute the difference between the short-term and long-term EMAs
    df['EMA_Diff'] = df['EMA_10'] - df['EMA_50']
    
    # Calculate the average true range (ATR) as a measure of volatility over 14 days
    df['tr'] = df['high'] - df['low']
    df['tr'] = df[['tr', (df['high'] - df['close'].shift()).abs(), (df['close'].shift() - df['low']).abs()]].max(axis=1)
    df['atr'] = df['tr'].rolling(window=14).mean()
    
    # Generate the final heuristic by combining the EMA_Diff and ATR with weights
    heuristics_matrix = 0.6 * df['EMA_Diff'] / df['atr'] + 0.4 * (df['close'] - df['EMA_10']) / df['atr']
    
    return heuristics_matrix
