def heuristics_v2(df):
    # Calculate 21-day momentum
    momentum_21 = df['close'] - df['close'].shift(21)
    
    # Calculate True Range
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift())
    tr3 = abs(df['low'] - df['close'].shift())
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate 50-day SMA of True Range
    sma_tr_50 = true_range.rolling(window=50).mean()
    
    # Create the heuristic matrix
    heuristics_matrix = (momentum_21 / sma_tr_50).fillna(0)
    return heuristics_matrix
