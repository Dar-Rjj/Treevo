import pandas as pd

def heuristics_v2(df):
    def compute_moving_averages(data, window):
        return data.rolling(window=window).mean()
    
    def compute_adjusted_high_low_ratio(data):
        return (data['high'] - data['open']) / (data['open'] - data['low'])
    
    def compute_momentum(data, window):
        return data['close'].pct_change(periods=window)
    
    def compute_volatility(data, window):
        return data.rolling(window=window).apply(lambda x: (x - x.mean()).abs().mean(), raw=True)
    
    short_window = 15
    long_window = 80
    momentum_medium_window = 30
    
    ma_short = compute_moving_averages(df['close'], short_window)
    ma_long = compute_moving_averages(df['close'], long_window)
    hlr_adjusted_ratio = compute_adjusted_high_low_ratio(df)
    momentum = compute_momentum(df, momentum_medium_window)
    volatility = compute_volatility(df['close'], short_window)
    
    heuristics_matrix = (ma_short - ma_long) * 0.4 + hlr_adjusted_ratio * 0.3 + momentum * 0.2 + volatility * 0.1
    return heuristics_matrix
