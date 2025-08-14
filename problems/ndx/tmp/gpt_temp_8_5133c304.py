import pandas as pd
import ta

def heuristics_v2(df):
    price_avg = (df['high'] + df['low']) / 2
    volume_sqrt_adj = price_avg * df['volume'] ** 0.5
    rsi = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    heuristics_matrix = 0.5 * volume_sqrt_adj + 0.5 * rsi
    return heuristics_matrix
