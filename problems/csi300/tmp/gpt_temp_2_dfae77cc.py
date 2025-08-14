import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Compute Intraday High-Low Spread
    df['Intraday_High_Low_Spread'] = df['High'] - df['Low']
    
    # Compute Previous Day's Close-to-Open Return
    df['Prev_Close_to_Open_Return'] = df['Close'].shift(1) - df['Open']
    
    # Compute Volume Weighted Average Price (VWAP)
    df['Price_Volume'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4 * df['Volume']
    df['Total_Price_Volume'] = df['Price_Volume'].cumsum()
    df['Total_Volume'] = df['Volume'].cumsum()
    df['VWAP'] = df['Total_Price_Volume'] / df['Total_Volume']
    
    # Combine Intraday Momentum and VWAP
    df['Combined_Value'] = df['VWAP'] - df['Intraday_High_Low_Spread']
    
    # Weight by Intraday Volume
    df['Weighted_Combined_Value'] = df['Combined_Value'] * df['Volume']
    
    # Smooth the Factor using Exponential Moving Average (EMA)
    df['Smoothed_Factor'] = df['Weighted_Combined_Value'].ewm(span=5, adjust=False).mean()
    
    return df['Smoothed_Factor']
