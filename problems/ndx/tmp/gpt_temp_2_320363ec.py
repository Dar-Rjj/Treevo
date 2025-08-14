import pandas as pd

def heuristics_v2(df):
    def calculate_roc(series, period=20):
        return (series / series.shift(period) - 1) * 100
    
    def calculate_vwap(high, low, close, volume):
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        return vwap
    
    def calculate_ema(series, span=15):
        return series.ewm(span=span, adjust=False).mean()

    roc = calculate_roc(df['close'])
    vwap = calculate_vwap(df['high'], df['low'], df['close'], df['volume'])
    ema_vwap = calculate_ema(vwap)
    
    heuristic_factor = df['close'] - ema_vwap
    return heuristics_matrix
