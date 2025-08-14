import pandas as pd

def heuristics_v2(df):
    # Calculate RSI
    delta = df['close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate ROC
    roc = df['close'].pct_change(periods=12)
    
    # Calculate the price-volume relationship
    pv_relationship = df['close'] / df['volume']
    
    # Combine RSI, ROC, and price-volume relationship
    combined_factors = (rsi + roc + pv_relationship).rank(pct=True)
    
    return heuristics_matrix
