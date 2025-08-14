def heuristics_v2(df):
    # Exponential Moving Average (EMA)
    ema_9 = df['close'].ewm(span=9, adjust=False).mean()
    
    # Historical Volatility (using standard deviation of log returns over 21 days)
    log_returns = np.log(df['close'] / df['close'].shift(1))
    hist_vol = log_returns.rolling(window=21).std() * np.sqrt(252)
    
    # Volume Rate of Change (VROC)
    vroc_14 = (df['volume'] - df['volume'].shift(14)) / df['volume'].shift(14) * 100
    
    # Composite heuristic
    heuristics_matrix = (ema_9 + hist_vol + vroc_14) / 3
    return heuristics_matrix
