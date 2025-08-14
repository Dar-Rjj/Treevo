import pandas as pd
import ta

def heuristics_v2(df):
    df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
    df['ma_close_20'] = df['close'].rolling(window=20).mean()
    df['high_ma_ratio'] = df['high'] / df['ma_close_20']
    heuristics_matrix = (df['rsi'] + df['high_ma_ratio']).dropna()
    return heuristics_matrix
