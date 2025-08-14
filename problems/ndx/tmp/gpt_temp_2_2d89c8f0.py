def heuristics_v2(df):
    # Calculate ROC
    roc = df['close'].pct_change(periods=10)
    
    # Calculate True Range
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate +DM and -DM
    dm_pos = (df['high'].diff(1).fillna(0)).where((df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']), 0)
    dm_neg = (df['low'].shift(1) - df['low']).where((df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)), 0)
    
    # Smooth +DM and -DM
    smooth_dm_pos = dm_pos.rolling(window=14).sum()
    smooth_dm_neg = dm_neg.rolling(window=14).sum()
    smooth_true_range = true_range.rolling(window=14).sum()
    
    # Calculate +DI and -DI
    di_pos = (smooth_dm_pos / smooth_true_range) * 100
    di_neg = (smooth_dm_neg / smooth_true_range) * 100
    
    # Calculate ATR
    atr = true_range.rolling(window=14).mean()
    
    # Composite heuristic
    heuristics_matrix = roc + (di_pos - di_neg) + atr
    
    return heuristics_matrix
