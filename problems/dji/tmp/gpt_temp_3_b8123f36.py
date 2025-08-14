def heuristics_v2(df):
    # Calculate the relative strength index (RSI) over the last 14 days
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate the average true range (ATR) over the last 14 days
    tr = df[['high', 'low']].apply(lambda x: np.max(np.abs(x - x.shift())), axis=1)
    atr = tr.rolling(window=14).mean()
    
    # Combine RSI and ATR into a single heuristics score
    heuristics_matrix = 0.7 * rsi + 0.3 * atr
    
    return heuristics_matrix
