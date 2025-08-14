def heuristics_v2(df):
    window_length = 14
    rsi_window = 5

    def compute_rsi(prices, window_length):
        diff = prices.diff(1)
        up_chg = 0 * diff
        down_chg = 0 * diff
    
        up_chg[diff > 0] = diff[diff > 0]
        down_chg[diff < 0] = diff[diff < 0]
    
        up_chg_avg = up_chg.rolling(window_length).mean()
        down_chg_avg = down_chg.rolling(window_length).abs().mean()
    
        rs = abs(up_chg_avg / down_chg_avg)
        rsi = 100 - 100 / (1 + rs)
        
        return rsi

    rsi = compute_rsi(df['close'], window_length)
    heuristics_matrix = rsi.rolling(window=rsi_window).mean().dropna()

    return heuristics_matrix
