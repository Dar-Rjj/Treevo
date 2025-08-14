import pandas as pd
import pandas as pd
import talib

def heuristics_v2(df):
    # Compute 5-day and 20-day simple moving averages (SMA) for the closing prices
    df['SMA_5'] = df['close'].rolling(window=5).mean()
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    
    # Create an alpha factor as the ratio of the 5-day SMA to the 20-day SMA
    df['SMA_Ratio'] = df['SMA_5'] / df['SMA_20']
    
    # Define a 20-day price return as the percentage change in the closing price from 20 days ago
    df['Price_Return_20'] = df['close'].pct_change(periods=20)
    
    # Construct a 10-day average direction index (ADX) using high and low prices to measure trend strength
    df['ADX_10'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=10)
    
    # Formulate a 20-day average daily trading volume
    df['Avg_Volume_20'] = df['volume'].rolling(window=20).mean()
    
    # Develop an alpha factor as the difference between the current day's volume and the 20-day average daily trading volume
    df['Volume_Diff_20'] = df['volume'] - df['Avg_Volume_20']
    
    # Calculate the 20-day cumulative volume delta, which is the sum of the differences between up-volume and down-volume over the last 20 days
    df['Volume_Delta'] = df['volume'] * (df['close'] > df['close'].shift(1)).astype(int) - df['volume'] * (df['close'] < df['close'].shift(1)).astype(int)
    df['Cum_Volume_Delta_20'] = df['Volume_Delta'].rolling(window=20).sum()
    
    # Generate a 20-day money flow index (MFI) by considering typical price and raw volume
    df['Typical_Price'] = (df['high'] + df['low'] + df['close']) / 3
    positive_money_flow = df['Typical_Price'] * df['volume'] * (df['Typical_Price'] > df['Typical_Price'].shift(1))
    negative_money_flow = df['Typical_Price'] * df['volume'] * (df['Typical_Price'] < df['Typical_Price'].shift(1))
    df['Positive_Money_Flow'] = positive_money_flow.rolling(window=20).sum()
    df['Negative_Money_Flow'] = negative_money_flow.rolling(window=20).sum()
    df['MFI_20'] = 100 - (100 / (1 + (df['Positive_Money_Flow'] / df['Negative_Money_Flow'])))
    
    # Measure the 20-day accumulation/distribution line (A/D Line) to assess the buying and selling pressure
    df['A/D_Line'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']) * df['volume']
    df['A/D_Line_Change_20'] = df['A/D_Line'].diff().rolling(window=20).sum()
    
    # Calculate the 20-day average of the difference between open and close prices
    df['Open_Close_Diff'] = df['open'] - df['close']
    df['Avg_Open_Close_Diff_20'] = df['Open_Close_Diff'].rolling(window=20).mean()
    
    # Determine the 20-day relative strength (RS) of the close price compared to the open price
    df['RS_20'] = (df['close'] / df['open']).rolling(window=20).mean()
    
    # Create a combined alpha factor
    df['Alpha_Factor'] = (df['SMA_Ratio'] + df['Price_Return_20'] + df['ADX_10'] + df['Volume_Diff_20'] + df['Cum_Volume_Delta_20'] + df['MFI_20'] + df['A/D_Line_Change_20'] + df['Avg_Open_Close_Diff_20'] + df['RS_20']).fillna(0)
    
    return df['Alpha_Factor']
