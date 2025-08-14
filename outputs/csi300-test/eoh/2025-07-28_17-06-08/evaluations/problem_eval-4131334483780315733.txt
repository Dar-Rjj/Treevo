import pandas as pd
import ta

def heuristics_v2(df):
    df['std_close_20'] = df['close'].rolling(window=20).std()
    df['rsi_14'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    heuristics_matrix = df['std_close_20'] * df['rsi_14']
    return heuristics_matrix
