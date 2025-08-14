import pandas as pd

def heuristics_v2(df):
    def compute_exponential_moving_average(data, span):
        return data.ewm(span=span, adjust=False).mean()
    
    def compute_rsi(data, window):
        delta = data.diff(1)
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        roll_up = up.ewm(alpha=1/window, adjust=False).mean()
        roll_down = down.abs().ewm(alpha=1/window, adjust=False).mean()
        rs = roll_up / roll_down
        return 100.0 - (100.0 / (1.0 + rs))
    
    def compute_custom_volatility(data, window):
        log_returns = data.apply(lambda x: np.log(x) - np.log(x.shift(1)))
        return log_returns.rolling(window=window).std()
    
    short_ema_span = 12
    long_ema_span = 26
    rsi_window = 14
    volatility_window = 20
    
    ema_short = compute_exponential_moving_average(df['close'], short_ema_span)
    ema_long = compute_exponential_moving_average(df['close'], long_ema_span)
    rsi = compute_rsi(df['close'], rsi_window)
    volatility = compute_custom_volatility(df['close'], volatility_window)
    
    heuristics_matrix = ema_short - ema_long + rsi + volatility
    return heuristics_matrix
