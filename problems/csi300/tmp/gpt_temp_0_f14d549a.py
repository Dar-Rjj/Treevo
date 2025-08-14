def heuristics_v2(df):
    # Calculate Volume-Weighted Moving Averages (VWMA)
    df['VWMA_5'] = (df['close'] * df['volume']).rolling(window=5).sum() / df['volume'].rolling(window=5).sum()
    df['VWMA_20'] = (df['close'] * df['volume']).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
    df['VWMA_60'] = (df['close'] * df['volume']).rolling(window=60).sum() / df['volume'].rolling(window=60).sum()

    # Calculate Exponential Moving Averages (EMA)
    df['EMA_5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['EMA_60'] = df['close'].ewm(span=60, adjust=False).mean()

    # Calculate Relative Strength Index (RSI) with Exponential Smoothing
    delta = df['close'].diff(1)
    gain = (delta.where(delta > 0, 0)).ewm(span=5, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(span=5, adjust=False).mean()
    rs = gain / loss
    df['RSI_5'] = 100 - (100 / (1 + rs))

    gain = (delta.where(delta > 0, 0)).ewm(span=14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(span=14, adjust=False).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # Calculate Volume-Weighted Average Price (VWAP)
    df['VWAP_5'] = (df['amount'] / df['volume']).rolling(window=5).mean()
    df['VWAP_20'] = (df['amount'] / df['volume']).rolling(window=20).mean()

    # Calculate Volatility Metrics
    df['std_returns_20'] = df['close'].pct_change().rolling(window=20).std()
