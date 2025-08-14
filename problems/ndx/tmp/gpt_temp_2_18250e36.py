import pandas as pd

def heuristics_v2(df):
    def compute_log_return(data, window):
        return (data['close'].pct_change(periods=window) + 1).apply(np.log)
    
    def compute_volume_ratio(data, window):
        avg_volume = data['volume'].rolling(window=window).mean()
        std_volume = data['volume'].rolling(window=window).std()
        return avg_volume / std_volume
    
    def compute_rsi(data, window):
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    log_return_window = 30
    volume_ratio_window = 20
    rsi_window = 14
    
    log_return = compute_log_return(df, log_return_window)
    volume_ratio = compute_volume_ratio(df, volume_ratio_window)
    rsi = compute_rsi(df, rsi_window)
    
    heuristics_matrix = log_return * 0.4 + volume_ratio * 0.3 + rsi * 0.3
    return heuristics_matrix
