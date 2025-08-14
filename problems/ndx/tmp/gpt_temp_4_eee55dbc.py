def heuristics_v2(df):
    def calc_adx(df, window=14):
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr = pd.DataFrame(index=df.index)
        tr['H-L'] = abs(high - low)
        tr['H-PC'] = abs(high - close.shift(1))
        tr['L-PC'] = abs(low - close.shift(1))
        tr['TR'] = tr.max(axis=1)
        
        atr = tr['TR'].rolling(window=window).mean()
        
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        pos_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        neg_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
        
        pos_di = 100 * (pos_dm.rolling(window=window).sum() / atr)
        neg_di = 100 * (neg_dm.rolling(window=window).sum() / atr)
        
        dx = 100 * (abs(pos_di - neg_di) / (pos_di + neg_di))
        adx = dx.rolling(window=window).mean()
        
        return adx
    
    def calc_volume_flow(df, window=20):
        return (df['close'] - df['close'].shift(1)) / (df['volume'].rolling(window=window).mean())
    
    adx = calc_adx(df)
    volume_flow = calc_volume_flow(df)
    
    heuristics_matrix = adx + volume_flow
    return heuristics_matrix
