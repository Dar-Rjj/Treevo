def heuristics_v2(df):
    # Momentum-Based Factors
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    df['SMA_200'] = df['close'].rolling(window=200).mean()
    df['SMA_Crossover'] = df['SMA_50'] - df['SMA_200']
    
    df['ROC'] = df['close'].pct_change(periods=12)
    
    df['EMA_5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['EMA_10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['Short_Term_Momentum'] = df['EMA_5'] - df['EMA_10']
    
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['Long_Term_Momentum'] = df['EMA_5'] - df['EMA_20']
    
    # Reversal-Based Factors
    def rsi(data, window=14):
        diff = data.diff(1)
        up = diff.where(diff > 0, 0.0)
        down = -diff.where(diff < 0, 0.0)
        avg_gain = up.rolling(window=window).mean()
        avg_loss = down.rolling(window=window).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
