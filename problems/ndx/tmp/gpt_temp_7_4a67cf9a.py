import pandas as pd

def heuristics_v2(df):
    daily_log_returns = df['close'].pct_change().apply(lambda x: np.log(1+x))
    std_dev = daily_log_returns.rolling(window=21).std()
    min_close_40d = df['close'].rolling(window=40).min()
    price_ratio = df['close'] / min_close_40d
    heuristics_matrix = 0.5 * std_dev + 0.5 * price_ratio
    return heuristics_matrix
