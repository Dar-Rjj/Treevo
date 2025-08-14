import pandas as pd

def heuristics_v2(df):
    def compute_log_diff(series_high, series_low, window):
        log_high = series_high.rolling(window=window).apply(lambda x: np.log(x[-1]))
        log_low = series_low.rolling(window=window).apply(lambda x: np.log(x[-1]))
        return log_high - log_low

    heuristics_matrix = compute_log_diff(df['high'], df['low'], 20)
    heuristics_matrix.name = 'heuristic_factor'
    return heuristics_matrix
