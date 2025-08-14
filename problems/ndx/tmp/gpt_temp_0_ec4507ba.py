def heuristics_v2(df):
    # Calculate logarithmic returns
    log_returns = np.log(df['close']).diff()
    
    # Calculate 50-day SMA
    sma_50 = df['close'].rolling(window=50).mean()
    
    # Calculate 14-day ADOSC
    ad = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']) * df['volume']
    adosc = ad.rolling(window=14).mean() - ad.rolling(window=3).mean()
    
    # Composite heuristic
    heuristics_matrix = log_returns + sma_50 + adosc
    
    return heuristics_matrix
