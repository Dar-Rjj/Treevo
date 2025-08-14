def heuristics_v2(df):
    # Trend Following Factors
    df['SMA_5'] = df['close'].rolling(window=5).mean()
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_Crossover'] = df['SMA_5'] - df['SMA_20']
    
    df['EMA_5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['EMA_Crossover'] = df['EMA_5'] - df['EMA_20']
    
    df['Close_minus_SMA_20'] = df['close'] - df['SMA_20']

    # Momentum Indicators
    def rsi(series, period):
        delta = series.diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
