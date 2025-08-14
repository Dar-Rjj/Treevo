import pandas as pd
import talib

def heuristics_v2(df):
    close = df['close']
    high = df['high']
    low = df['low']

    # Calculate the rate of change in closing prices over 20 days
    roc = talib.ROC(close, timeperiod=20)

    # Calculate the ADX over 14 days
    adx = talib.ADX(high, low, close, timeperiod=14)

    # Combine ROC and ADX
    heuristics_matrix = roc + adx

    return heuristics_matrix
