def heuristics_v2(df):
    # Momentum-Based Factors
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    df['SMA_200'] = df['close'].rolling(window=200).mean()
    df['Momentum_SMA_Crossover'] = df['SMA_50'] - df['SMA_200']
    
    df['ROC_12'] = df['close'].pct_change(periods=12)
    
    df['EMA_5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['EMA_10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['Short_Term_Momentum'] = df['EMA_5'] - df['EMA_10']
    
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['Long_Term_Momentum'] = df['EMA_5'] - df['EMA_20']

    # Reversal-Based Factors
    def rsi(series, period=14):
        delta = series.diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
