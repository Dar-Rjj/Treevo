def heuristics_v2(df):
    diff = df['high'] - df['low']
    rsi_period = 14
    delta = diff.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    smoothed_rsi = rsi.ewm(alpha=0.3).mean().dropna()
    
    return heuristics_matrix
