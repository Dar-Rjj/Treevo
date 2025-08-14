def heuristics_v2(df):
    atr_period = 14
    roc_period = 10
    sma_period = 5
    
    tr = pd.DataFrame({
        'h-l': df['high'] - df['low'],
        'h-pc': abs(df['high'] - df['close'].shift(1)),
        'l-pc': abs(df['low'] - df['close'].shift(1))
    }).max(axis=1)
    
    atr = tr.rolling(window=atr_period).mean()
    roc = ((atr / atr.shift(roc_period)) - 1) * 100
    heuristics_matrix = roc.rolling(window=sma_period).mean().dropna()
    
    return heuristics_matrix
