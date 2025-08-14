def heuristics_v2(df):
    # Simple Moving Averages
    df['SMA_5'] = df['close'].rolling(window=5).mean()
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    
    # Exponential Moving Averages
    df['EMA_5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
    
    # Daily Price Change
    df['Daily_Change'] = df['close'].diff()
    
    # High-Low Spread
    df['High_Low_Spread'] = df['high'] - df['low']
    
    # Open-Close Spread
    df['Open_Close_Spread'] = df['open'] - df['close']
    
    # Rate of Change (ROC)
    df['ROC_5'] = df['close'].pct_change(periods=5)
    df['ROC_20'] = df['close'].pct_change(periods=20)
    
    # Relative Strength Index (RSI)
    def compute_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
