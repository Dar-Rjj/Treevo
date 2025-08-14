import pandas as pd

def heuristics_v2(df):
    def compute_exponential_moving_averages(data, window):
        return data.ewm(span=window, adjust=False).mean()
    
    def compute_log_return(data, window):
        return (data['close'] / data['close'].shift(window)).apply(lambda x: math.log(x))
    
    def compute_liquidity(data):
        return data['volume'] / data['close']
    
    short_window = 10
    long_window = 50
    log_return_window = 10
    
    ema_short = compute_exponential_moving_averages(df['close'], short_window)
    ema_long = compute_exponential_moving_averages(df['close'], long_window)
    log_return = compute_log_return(df, log_return_window)
    liquidity = compute_liquidity(df)
    
    heuristics_matrix = (ema_short - ema_long) * 0.4 + log_return * 0.3 + liquidity * 0.3
    return heuristics_matrix
