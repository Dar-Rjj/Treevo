import pandas as pd

def heuristics_v2(df):
    def compute_ema(data, span):
        return data.ewm(span=span, adjust=False).mean()
    
    def compute_daily_return(data):
        return data['close'].pct_change().fillna(0)
    
    ema_short_span = 20
    ema_long_span = 80
    momentum_period = 15
    
    ema_short = compute_ema(df['close'], ema_short_span)
    ema_long = compute_ema(df['close'], ema_long_span)
    daily_return = compute_daily_return(df)
    momentum = daily_return.rolling(window=momentum_period).mean()
    volatility = daily_return.rolling(window=momentum_period).std()
    
    heuristics_matrix = (ema_short / ema_long) + momentum * (1 - (volatility / volatility.mean()))
    return heuristics_matrix
