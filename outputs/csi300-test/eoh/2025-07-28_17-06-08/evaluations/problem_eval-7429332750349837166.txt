import pandas as pd

def heuristics_v2(df):
    def compute_ema(df, column, span=10):
        return df[column].ewm(span=span, adjust=False).mean()

    returns = df['close'].pct_change()
    ema_returns = compute_ema(returns, 'close', 10)
    ema_volume = compute_ema(df, 'volume', 10)
    
    heuristics_matrix = ema_returns * ema_volume
    return heuristics_matrix
