import pandas as pd
import numpy as np

def heuristics_v2(df):
    def ema_diff(price, fast=10, slow=50):
        ema_fast = price.ewm(span=fast, adjust=False).mean()
        ema_slow = price.ewm(span=slow, adjust=False).mean()
        return ema_fast - ema_slow
    
    def atr(df, period=14):
        tr = df[['high', 'low', 'close']].apply(lambda x: max(x['high'] - x['low'], abs(x['high'] - x['close'].shift(1)), abs(x['low'] - x['close'].shift(1))), axis=1)
        return tr.ewm(span=period, adjust=False).mean()

    ema_signal = ema_diff(df['close'])
    atr_value = atr(df)
    combined_factor = (ema_signal + atr_value).rename('combined_factor')
    heuristics_matrix = combined_factor.ewm(span=30, adjust=False).mean().rename('heuristic_factor')

    return heuristics_matrix
