import pandas as pd
import talib

def heuristics_v2(df):
    # Exponential Moving Averages
    ema_10 = df['close'].ewm(span=10, adjust=False).mean()
    ema_30 = df['close'].ewm(span=30, adjust=False).mean()
    ema_ratio = (ema_10 / ema_30) - 1
    
    # Average Directional Index (ADX)
    adx = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
    
    # Price Rate of Change (ROC)
    roc = talib.ROCP(df['close'], timeperiod=10)
    
    # Price range
    price_range = (df['high'] - df['low']) / df['close']
    
    # Volume weighted by close price
    volume_weighted = df['volume'] * df['close']
    
    # Composite heuristic
    heuristics_matrix = (ema_ratio + adx + roc + price_range + volume_weighted) / 5
    return heuristics_matrix
