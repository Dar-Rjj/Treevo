def heuristics_v2(df):
    # Calculate 14-day RSI
    delta = df['close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 14-day ATR
    tr = pd.DataFrame({'H-L': df['high'] - df['low'], 'H-PC': abs(df['high'] - df['close'].shift(1)), 'L-PC': abs(df['low'] - df['close'].shift(1))})
    tr['TR'] = tr.max(axis=1)
    atr = tr['TR'].rolling(window=14).mean()
    
    # Calculate 10-day moving average of the closing price
    close_ma = df['close'].rolling(window=10).mean()
    
    # Generate the heuristics factor
    df['Heuristic_Factor'] = (rsi / atr) * close_ma
    
    heuristics_matrix = df['Heuristic_Factor'].dropna()
    return heuristics_matrix
