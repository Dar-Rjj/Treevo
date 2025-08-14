def heuristics_v2(df):
    atr_window = 14
    std_window = atr_window
    sma_window = 7

    tr = df['high'] - df['low']
    tr['h-l'] = tr
    tr['h-yc'] = (df['high'] - df['close'].shift(1)).abs()
    tr['l-yc'] = (df['low'] - df['close'].shift(1)).abs()
    tr['tr'] = tr[['h-l', 'h-yc', 'l-yc']].max(axis=1)
    atr = tr['tr'].rolling(window=atr_window).mean()

    std_close = df['close'].rolling(window=std_window).std()
    
    heuristics_matrix = (atr / std_close).rolling(window=sma_window).mean().dropna()
    
    return heuristics_matrix
