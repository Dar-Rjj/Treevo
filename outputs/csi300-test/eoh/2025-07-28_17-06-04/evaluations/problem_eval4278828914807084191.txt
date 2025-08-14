import pandas as pd
import talib

def heuristics_v2(df):
    # Calculate the RSI of the close price
    rsi = talib.RSI(df['close'], timeperiod=14)
    # Adjust the rate of change of volume with the RSI
    roc_volume_adjusted = df['volume'].pct_change() * (rsi / 100)
    # Calculate the True Range
    tr = df['high'] - df['low']
    # Calculate the Exponential Moving Average of the True Range
    ema_tr = talib.EMA(tr, timeperiod=14)
    # Combine the two adjusted factors to create a new heuristics matrix
    heuristics_matrix = (roc_volume_adjusted + ema_tr).dropna()
    return heuristics_matrix
