def heuristics_v2(df):
    # Calculate the price change
    delta = df['close'].diff().fillna(0)
    
    # Calculate the average true range (ATR)
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift())
    tr3 = abs(df['low'] - df['close'].shift())
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=14).mean()
    
    # Calculate the relative strength (RS) of price changes
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    
    # Calculate the weighted combination of momentum and volatility
    heuristics_matrix = (rs * delta + atr).rolling(window=5).mean()
    
    return heuristics_matrix
