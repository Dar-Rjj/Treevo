import pandas as pd

def heuristics_v2(df):
    def calculate_moving_average(data, window):
        return data.rolling(window=window).mean()
    
    def calculate_rsi(data, window):
        delta = data.diff(1)
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        roll_up = up.rolling(window).mean()
        roll_down = down.abs().rolling(window).mean()
        rs = roll_up / roll_down
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi
    
    def calculate_roc(data, window):
        return (data / data.shift(window)) - 1
    
    short_window = 14
    long_window = 30
    momentum_window = 10
    
    ma_short = df['close'].apply(lambda x: calculate_moving_average(x, short_window))
    ma_long = df['close'].apply(lambda x: calculate_moving_average(x, long_window))
    rsi = df['close'].apply(lambda x: calculate_rsi(x, 14))
    roc = df['close'].apply(lambda x: calculate_roc(x, 9))
    momentum = df['close'].pct_change(momentum_window)
    
    # Combine factors
    heuristics_matrix = (ma_short - ma_long) + rsi + roc + momentum
    return heuristics_matrix
