def heuristics_v2(df):
    # Calculate the 20-day Rate of Change (ROC)
    roc = (df['close'] - df['close'].shift(20)) / df['close'].shift(20) * 100
    
    # Calculate the 5-day Weighted Moving Average (WMA) of volume
    weights = pd.Series([1, 2, 3, 4, 5])
    wma_volume = (df['volume'] * weights).rolling(window=5).sum() / weights.sum()
    
    # Apply a custom heuristic to combine the ROC and WMA of volume
    heuristics_matrix = (roc + wma_volume).rank(pct=True)
    
    return heuristics_matrix
