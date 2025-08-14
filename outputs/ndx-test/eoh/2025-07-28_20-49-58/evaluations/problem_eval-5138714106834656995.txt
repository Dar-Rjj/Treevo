def heuristics_v2(df):
    def calculate_roc(column, n):
        return (column - column.shift(n)) / column.shift(n)
    
    # Calculate ROC over a 20-day period
    roc = calculate_roc(df['close'], 20)
    
    # Calculate 10-day exponential moving average of the closing price
    ema_10 = df['close'].ewm(span=10, adjust=False).mean()
    
    # Calculate VWAP with 10-day EMA
    tp = (df['high'] + df['low'] + ema_10) / 3
    vwap = (df['volume'] * tp).cumsum() / df['volume'].cumsum()
    
    # Capture the 10-day ATR
    tr = pd.Series(0.0, index=df.index)
    tr = pd.concat([tr, (df['high'] - df['low']).rename('hl'), 
                    (df['high'] - df['close'].shift()).abs().rename('hc'),
                    (df['low'] - df['close'].shift()).abs().rename('lc')], axis=1).max(axis=1)
    atr_10 = tr.rolling(window=10).mean()
    
    # Adjust the heuristics matrix with momentum and volatility
    heuristics_matrix = roc + (vwap - df['close']) + roc.rolling(window=5).mean() - atr_10
    
    return heuristics_matrix
