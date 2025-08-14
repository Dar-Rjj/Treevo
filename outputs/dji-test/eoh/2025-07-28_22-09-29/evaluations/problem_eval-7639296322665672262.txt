import pandas as pd
import ta

def heuristics_v2(df):
    def rsi(price, window=14):
        return ta.momentum.RSIIndicator(price, window).rsi()

    def roc(series, periods=30):
        return series.pct_change(periods=periods)

    rsi_signal = rsi(df['close'])
    roc_low = roc(df['low'])
    combined_factor = (rsi_signal + roc_low).rename('combined_factor')
    heuristics_matrix = combined_factor.rolling(window=10).mean().rename('heuristic_factor')

    return heuristics_matrix
