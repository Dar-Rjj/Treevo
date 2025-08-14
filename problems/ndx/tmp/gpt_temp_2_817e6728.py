import pandas as pd

def heuristics_v2(df):
    def compute_moving_averages(data, window):
        return data.rolling(window=window).mean()

    def compute_price_to_volume_ratio(data):
        return (data['close'] - data['open']) / data['volume']

    def compute_log_momentum(data, window):
        return (data['close'].apply(np.log) - data['close'].shift(window).apply(np.log))

    def compute_volatility(data, window):
        return data['close'].pct_change().rolling(window=window).std()

    def compute_close_to_avg_high_low(data):
        return (data['close'] - (data['high'] + data['low']) / 2)

    short_window = 10
    long_window = 50
    momentum_window = 30
    volatility_window = 20

    ma_short = compute_moving_averages(df['close'], short_window)
    ma_long = compute_moving_averages(df['close'], long_window)
    ptv_ratio = compute_price_to_volume_ratio(df)
    log_momentum = compute_log_momentum(df, momentum_window)
    volatility = compute_volatility(df, volatility_window)
    close_to_avg = compute_close_to_avg_high_low(df)

    heuristics_matrix = (ma_short - ma_long) + (ptv_ratio * log_momentum) + (close_to_avg / volatility)
    return heuristics_matrix
