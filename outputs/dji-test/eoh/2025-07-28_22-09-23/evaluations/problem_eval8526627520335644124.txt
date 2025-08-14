import pandas as pd

def heuristics_v2(df):
    def ema(series, span=20):
        return series.ewm(span=span, adjust=False).mean()
    
    def rsi(series, n=20):
        delta = series.diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=n).min_periods(1).mean()
        avg_loss = loss.rolling(window=n).min_periods(1).mean()
        rs = avg_gain / avg_loss
        rsi_series = 100 - (100 / (1 + rs))
        return rsi_series
    
    close_ema = ema(df['close'])
    close_rsi = rsi(df['close'])
    heuristics_matrix = (close_ema + close_rsi) / 2
    return heuristics_matrix
