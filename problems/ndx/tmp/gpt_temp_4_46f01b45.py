def heuristics_v2(df):
    def compute_atr(data, window):
        high_low = data['high'] - data['low']
        high_close = abs(data['high'] - data['close'].shift())
        low_close = abs(data['low'] - data['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = ranges.rolling(window=window).mean()
        return atr
    
    def compute_dmi(data, window):
        up_move = data['high'].diff(1)
        down_move = data['low'].diff(1)
        
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
        
        tr = compute_atr(data, 1)
        smooth_plus_dm = plus_dm.ewm(span=window, adjust=False).mean()
        smooth_minus_dm = minus_dm.ewm(span=window, adjust=False).mean()
        
        di_plus = 100 * (smooth_plus_dm / tr)
        di_minus = 100 * (smooth_minus_dm / tr)
        
        dmi = di_plus - di_minus
        return dmi

    atr_window = 14
    dmi_window = 14
    
    atr = compute_atr(df, atr_window)
    dmi = compute_dmi(df, dmi_window)
    
    heuristics_matrix = atr + dmi
    return heuristics_matrix
