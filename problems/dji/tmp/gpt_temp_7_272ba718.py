def heuristics_v2(df):
    # Calculate mPPUV (Modified Price Per Unit Volume)
    df['mPPUV'] = (df['close'] - df['open']) / (df['high'] - df['low']) * (df['close'] / df['volume'])
    
    # Calculate n-day relative strength (RS) factor
    df['close_change'] = df['close'].diff()
    df['pos_changes'] = df['close_change'].apply(lambda x: max(x, 0))
    df['neg_changes'] = df['close_change'].apply(lambda x: abs(min(x, 0)))
    window = 14  # Example window
    df['RS'] = df['pos_changes'].rolling(window=window).sum() / df['neg_changes'].rolling(window=window).sum()
    
    # Calculate smoothed AHLR (sAHLR)
    m = 10  # Example smoothing window
    df['AHLR'] = (df['high'] - df['low']) / df['close']
    df['sAHLR'] = df['AHLR'].rolling(window=m).mean()
    
    # Calculate volume stability factor
    df['volatility'] = (df['high'] - df['low']).rolling(window=20).std()
    df['stability_factor'] = 1 / df['volume'].rolling(window=20).std()
    
    # Calculate Enhanced Intraday Momentum
    df['high_low_diff'] = df['high'] - df['low']
    df['open_close_return'] = df['close'] / df['open'] - 1
