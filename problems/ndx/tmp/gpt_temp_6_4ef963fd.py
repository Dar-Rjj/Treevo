import pandas as pd

def heuristics_v2(df):
    def compute_roc(data, window):
        return ((data - data.shift(window)) / data.shift(window)) * 100
    
    def compute_moving_averages(data, window):
        return data.rolling(window=window).mean()
    
    def compute_volume_ratio_log(data, window):
        avg_volume = data.rolling(window=window).mean()
        return (data / avg_volume).apply(np.log)

    roc_window = 20
    ma_short_window = 50
    ma_long_window = 200
    vol_avg_window = 120

    roc = compute_roc(df['close'], roc_window)
    ma_short = compute_moving_averages(df['close'], ma_short_window)
    ma_long = compute_moving_averages(df['close'], ma_long_window)
    vol_ratio_log = compute_volume_ratio_log(df['volume'], vol_avg_window)

    heuristics_matrix = 0.5 * roc + 0.3 * (ma_short - ma_long) + 0.2 * vol_ratio_log
    return heuristics_matrix
