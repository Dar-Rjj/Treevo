import pandas as pd
import ta

def heuristics_v2(df):
    rsi = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    macd_signal = ta.trend.MACD(df['close']).macd_signal()
    adjusted_rsi = rsi * df['volume'].rolling(window=5).mean()
    heuristics_matrix = 0.7 * adjusted_rsi + 0.3 * macd_signal
    return heuristics_matrix
