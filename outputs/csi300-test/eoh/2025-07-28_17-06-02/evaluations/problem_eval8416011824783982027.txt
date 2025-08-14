def heuristics_v2(df):
    # Percentage Change in Closing Price
    pct_change = df['close'].pct_change()
    
    # True Range
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift())
    tr3 = abs(df['low'] - df['close'].shift())
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Exponentially Weighted Moving Average (EWMA) of True Range with a span of 5 days
    ewma_true_range = true_range.ewm(span=5).mean()
    
    # Heuristics matrix combining percentage change and EWMA of TR
    heuristics_matrix = pct_change + ewma_true_range
    return heuristics_matrix
