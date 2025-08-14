def heuristics_v2(df):
    # Calculate momentum indicators
    df['SMA_10'] = df['close'].rolling(window=10).mean()
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
    
    def rsi(series, period=14):
        delta = series.diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
