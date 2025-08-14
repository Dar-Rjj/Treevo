import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Price Range and Open-Close Return
    df['Range'] = df['High'] - df['Low']
    df['Open_Close_Return'] = (df['Close'] - df['Open']) / df['Open']
    df['Intraday_Metric'] = (df['Range'] + df['Open_Close_Return']) / 2

    # Measure Volume Synchronization with Shock Filter
    df['Volume_Change'] = df['Volume'] - df['Volume'].shift(1)
    df['Price_Change'] = df['Close'] - df['Close'].shift(1)
    df['Volume_Ratio'] = df['Volume'] / df['Volume'].shift(1)
    df['Volume_Shock'] = df['Volume_Ratio'] > 1.5
    df['Synchronized_Change'] = df['Volume_Change'] * df['Price_Change']
    df['Synchronized_Sign'] = np.sign(df['Synchronized_Change'])

    # Calculate On-Balance Volume (OBV)
    obv = [0]
    for i in range(1, len(df)):
        if df['Close'][i] > df['Close'][i-1]:
            obv.append(obv[-1] + df['Volume'][i])
        elif df['Close'][i] < df['Close'][i-1]:
            obv.append(obv[-1] - df['Volume'][i])
        else:
            obv.append(obv[-1])
    df['OBV'] = obv

    # Calculate Moving Average Convergence Divergence (MACD)
    df['Short_EMA'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['Long_EMA'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_Line'] = df['Short_EMA'] - df['Long_EMA']
    df['Signal_Line'] = df['MACD_Line'].ewm(span=9, adjust=False).mean()

    # Determine Momentum
    df['Momentum'] = np.where(df['MACD_Line'] > df['Signal_Line'], 'Bullish', 'Bearish')

    # Weight OBV by the Difference between MACD Line and Signal Line
    df['Weight'] = abs(df['MACD_Line'] - df['Signal_Line'])
    df['OBV_Factor'] = df['OBV'] * df['Weight']

    # Combine Smoothed Intraday Metrics with OBV Factor
    df['Integrated_Metric'] = (df['Intraday_Metric'] * df['Open_Close_Return']) / df['Volume']
    df['Smoothed_Intraday'] = df['Integrated_Metric'] * df['OBV_Factor']

    # Final Alpha Factor
    df['Final_Alpha_Factor'] = (df['Intraday_Metric'] + df['Synchronized_Sign']) * df['OBV_Factor']

    return df['Final_Alpha_Factor']
