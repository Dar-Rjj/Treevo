import pandas as pd
import ta

def heuristics_v2(df):
    rsi = ta.momentum.RSIIndicator(df['close'], window=20).rsi()
    rsi_ma = rsi.rolling(window=10).mean()
    rsi_diff = (rsi - rsi_ma) / df['close'].std()
    
    macd_line = ta.trend.MACD(df['close']).macd_signal()
    volume_ratio = df['volume'] / df['volume'].rolling(window=30).mean()
    
    heuristics_matrix = rsi_diff + macd_line * volume_ratio
    return heuristics_matrix
