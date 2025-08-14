def heuristics_v2(df):
    # Simple Moving Averages (SMA)
    df['5D_Close_SMA'] = df['close'].rolling(window=5).mean()
    df['20D_Close_SMA'] = df['close'].rolling(window=20).mean()
    df['60D_Close_SMA'] = df['close'].rolling(window=60).mean()
    
    df['10D_High_SMA'] = df['high'].rolling(window=10).mean()
    df['30D_High_SMA'] = df['high'].rolling(window=30).mean()
    
    df['10D_Low_SMA'] = df['low'].rolling(window=10).mean()
    df['30D_Low_SMA'] = df['low'].rolling(window=30).mean()
    
    # Exponential Moving Averages (EMA)
    df['5D_Close_EMA'] = df['close'].ewm(span=5, adjust=False).mean()
    df['20D_Close_EMA'] = df['close'].ewm(span=20, adjust=False).mean()
    df['60D_Close_EMA'] = df['close'].ewm(span=60, adjust=False).mean()
    
    df['10D_High_EMA'] = df['high'].ewm(span=10, adjust=False).mean()
    df['30D_High_EMA'] = df['high'].ewm(span=30, adjust=False).mean()
    
    df['10D_Low_EMA'] = df['low'].ewm(span=10, adjust=False).mean()
    df['30D_Low_EMA'] = df['low'].ewm(span=30, adjust=False).mean()
    
    # Momentum Indicators
    df['10D_ROC'] = df['close'].pct_change(periods=10) * 100
    df['30D_ROC'] = df['close'].pct_change(periods=30) * 100
    
    def rsi(series, n=14):
        delta = series.diff(1)
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.rolling(window=n).mean()
        avg_loss = loss.rolling(window=n).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
