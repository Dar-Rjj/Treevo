import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Adjusted High-Low Spread
    df['High_Low_Spread'] = df['High'] - df['Low']
    
    # Weight by Volume
    df['Weighted_Spread'] = df['Volume'] * df['High_Low_Spread']
    
    # Condition on Close-to-Open Return
    df['Close_to_Open_Return'] = (df['Close'] - df['Open']) / df['Open']
    df['Positive_Return_Weight'] = df['Weighted_Spread'] * (df['Close_to_Open_Return'] > 0)
    df['Negative_Return_Weight'] = df['Weighted_Spread'] * (df['Close_to_Open_Return'] <= 0)
    
    # Calculate Intraday Percent Change
    df['Intraday_Percent_Change'] = (df['Close'] - df['Open']) / df['Open']
    
    # Integrate Volume Weighted Average Price
    df['VWAP'] = (df['High'] + df['Low'] + df['Close']) / 3 * df['Volume']
    df['VWAP'] = df['VWAP'].cumsum() / df['Volume'].cumsum()
    
    # Incorporate Previous Day's Close
    df['Prev_Close'] = df['Close'].shift(1)
    df['Prev_Open_Compare'] = (df['Open'] - df['Prev_Close']) / df['Prev_Close']
    
    # Enhance with Intraday Volume Ratio
    df['Prev_Volume'] = df['Volume'].shift(1)
    df['Intraday_Volume_Ratio'] = df['Volume'] / df['Prev_Volume']
    df['Final_Indicator'] = df['Intraday_Percent_Change'] + df['VWAP'] + df['Prev_Open_Compare']
    df['Final_Indicator'] = df['Final_Indicator'] * (df['Intraday_Volume_Ratio'] > 1).astype(int) * 1.5
    df['Final_Indicator'] = df['Final_Indicator'] * (df['Intraday_Volume_Ratio'] <= 1).astype(int) * 0.5
    
    # Calculate Daily Returns
    df['Daily_Returns'] = df['Close'].pct_change()
    
    # Short-Term EMA
    df['Short_EMA'] = df['Daily_Returns'].ewm(span=5, adjust=False).mean()
    
    # Long-Term EMA
    df['Long_EMA'] = df['Daily_Returns'].ewm(span=20, adjust=False).mean()
    
    # Compute Difference and Integrate with Combined Intraday Indicators
    df['EMA_Diff'] = df['Short_EMA'] - df['Long_EMA']
    df['Final_Alpha_Factor'] = df['Final_Indicator'] + df['EMA_Diff']
    
    return df['Final_Alpha_Factor'].dropna()
