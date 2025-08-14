import pandas as pd

def heuristics_v2(df):
    daily_log_return = (df['close'].pct_change() + 1).apply(np.log).fillna(0)
    cum_sum_daily_log_return_volume = (daily_log_return * df['volume']).cumsum()
    sma_cum_sum = cum_sum_daily_log_return_volume.rolling(window=5).mean()
    ema_close = df['close'].ewm(span=20, adjust=False).mean()
    heuristics_matrix = sma_cum_sum / ema_close
    return heuristics_matrix
