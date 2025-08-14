def heuristics_v2(df):
    # Momentum-based Alpha Factors
    df['SMA_5'] = df['close'].rolling(window=5).mean()
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['ROC_1'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1) * 100
    df['ROC_10'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10) * 100
    
    def rsi(series, period):
        delta = series.diff(1)
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
