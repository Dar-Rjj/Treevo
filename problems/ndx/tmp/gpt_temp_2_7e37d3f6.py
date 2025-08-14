import pandas as pd

def heuristics_v2(df):
    def compute_ema(data, span):
        return data.ewm(span=span, adjust=False).mean()
    
    def compute_daily_return(data):
        return data['close'].pct_change().fillna(0)
    
    ema_short_span = 12
    ema_long_span = 60
    momentum_period = 14
    
    ema_short = compute_ema(df['close'], ema_short_span)
    ema_long = compute_ema(df['close'], ema_long_span)
    daily_return = compute_daily_return(df)
    momentum = compute_ema(daily_return, momentum_period)
    volatility = daily_return.rolling(window=momentum_period).apply(lambda x: pd.Series(x).mad(), raw=True)
    
    heuristics_matrix = (ema_short - ema_long) + momentum * (1 - (volatility / volatility.median()))
    return heuristics_matrix
