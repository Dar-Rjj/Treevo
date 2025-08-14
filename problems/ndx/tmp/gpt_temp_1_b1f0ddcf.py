def heuristics_v2(df):
    # Calculate the accumulation/distribution line
    adl = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    adl = adl * df['volume']
    adl = adl.cumsum()
    
    # Calculate the true range
    tr = pd.DataFrame({'hl': df['high'] - df['low'], 'hc': abs(df['high'] - df['close'].shift(1)), 'lc': abs(df['low'] - df['close'].shift(1))})
    tr = tr.max(axis=1)
    
    # Calculate the average true range (ATR)
    atr = tr.rolling(window=14).mean()
    
    # Calculate the +DI
    dm_pos = (df['high'] - df['high'].shift(1))
    dm_neg = (df['low'].shift(1) - df['low'])
    dm_pos[dm_pos < 0] = 0
    dm_pos[(dm_pos > dm_neg) & (dm_neg > 0)] = 0
    di_pos = 100 * (dm_pos.rolling(window=14).sum() / (atr * 14))
    
    # Combine the factors into a composite heuristics measure
    heuristics_matrix = (adl + atr + di_pos) / 3
    return heuristics_matrix
