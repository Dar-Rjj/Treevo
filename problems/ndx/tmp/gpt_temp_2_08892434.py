import pandas as pd
import ta.momentum as mom

def heuristics_v2(df):
    price_diff = df['high'] - df['low']
    volume_adj = price_diff * df['volume']
    rsi = mom.rsi(df['close'], window=14)
    heuristics_matrix = 0.5 * volume_adj.rolling(window=7).std() + 0.5 * rsi
    return heuristics_matrix
