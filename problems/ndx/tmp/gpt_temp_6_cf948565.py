import pandas as pd
import talib

def heuristics_v2(df):
    adx = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
    rsi = talib.RSI(df['close'], timeperiod=14)
    hl_ratio = df['high'] / df['low']
    
    heuristics_matrix = 0.5 * adx + 0.3 * rsi + 0.2 * hl_ratio
    
    return heuristics_matrix
