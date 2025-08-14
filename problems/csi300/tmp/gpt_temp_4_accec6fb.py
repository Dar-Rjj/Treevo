import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from talib import EMA, RSI

def heuristics_v2(df):
    # Calculate High-Low Spread
    df['High_Low_Spread'] = df['high'] - df['low']

    # Calculate Daily Volume Trend
    df['Volume_MA_20'] = df['volume'].rolling(window=20).mean()
    df['Volume_Trend'] = df['volume'] - df['Volume_MA_20']
    df['Volume_Trend_Positive'] = df['Volume_Trend'] > 0

    # Calculate Price Trend
    df['Close_EMA_10'] = EMA(df['close'], timeperiod=10)
    df['Price_Trend'] = df['close'] - df['Close_EMA_10']
    df['Price_Trend_Positive'] = df['Price_Trend'] > 0

    # Calculate Momentum
    df['Return_30'] = (df['close'] / df['close'].shift(30)) - 1
    df['Momentum_Positive'] = df['Return_30'] > 0

    # Calculate Market Sentiment
    df['RSI_14'] = RSI(df['close'], timeperiod=14)
    df['Overbought'] = df['RSI_14'] > 70
    df['Oversold'] = df['RSI_14'] < 30

    # Combine Spread, Volume, Price Trends, Momentum, and Sentiment
    df['Adjusted_Spread'] = df.apply(lambda row: 
        (row['High_Low_Spread'] * 1.5 if row['Volume_Trend_Positive'] else row['High_Low_Spread'] * 0.5) *
        (1.2 if row['Price_Trend_Positive'] else 0.8) *
        (1.3 if row['Momentum_Positive'] else 0.7) *
        (0.6 if row['Overbought'] else 1.4 if row['Oversold'] else 1),
        axis=1
    )

    return df['Adjusted_Spread']
