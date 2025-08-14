import pandas as pd

def heuristics_v2(df):
    def compute_average_volume(data, window):
        return data.rolling(window=window).mean()
    
    def compute_rsi(data, window):
        delta = data.diff(1)
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        rma_up = up.ewm(alpha=1 / window, min_periods=window).mean()
        rma_down = down.abs().ewm(alpha=1 / window, min_periods=window).mean()
        rs = rma_up / rma_down
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def compute_log_change(data):
        return data.pct_change().fillna(0).apply(lambda x: np.log(x + 1))
    
    avg_volume_window = 10
    rsi_window = 14
    
    avg_volume = compute_average_volume(df['volume'], avg_volume_window)
    max_volume = df['volume'].rolling(window=avg_volume_window).max()
    volume_ratio = avg_volume / max_volume
    rsi = compute_rsi(df['close'], rsi_window)
    log_change = compute_log_change(df['close'])
    
    heuristics_matrix = volume_ratio * 0.5 + rsi * 0.3 + log_change * 0.2
    return heuristics_matrix
