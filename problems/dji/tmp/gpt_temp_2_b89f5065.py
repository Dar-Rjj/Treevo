import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Volume-Weighted Price
    df['Volume-Weighted_Price'] = (df['volume'] * df['close']).cumsum() / df['volume'].cumsum()
    
    # Calculate Daily Log Return
    df['Daily_Log_Return'] = np.log(df['close'] / df['close'].shift(1))
    
    # Calculate 20-Day Moving Average of Close Price
    df['20_Day_MA_Close'] = df['close'].rolling(window=20).mean()
    
    # Calculate 20-Day Standard Deviation of Log Returns
    df['20_Day_Std_Log_Return'] = df['Daily_Log_Return'].rolling(window=20).std()
    
    # Calculate Trend Momentum Indicator
    df['Trend_Momentum'] = (df['close'] - df['20_Day_MA_Close']) / df['20_Day_Std_Log_Return']
    
    # Sum Daily Returns over 5 days
    df['5_Day_Sum_Return'] = df['Daily_Log_Return'].rolling(window=5).sum()
    
    # Confirm Momentum with Volume and Amount
    df['Volume_Change'] = df['volume'] - df['volume'].shift(1)
    df['Amount_Change'] = df['amount'] - df['amount'].shift(1)
    
    volume_threshold = 0.1 * df['volume'].rolling(window=20).mean()
    amount_threshold = 0.1 * df['amount'].rolling(window=20).mean()
    
    df['Factor'] = 0
    condition = (df['Volume_Change'] > volume_threshold) & (df['Amount_Change'] > amount_threshold)
    df.loc[condition, 'Factor'] = df['Trend_Momentum']
    
    # Calculate Price Volatility
    df['High_Low_Range'] = df['high'] - df['low']
    df['Range_Std'] = df['High_Low_Range'].rolling(window=20).std()
    
    # Calculate Intraday High-Low Spread
    df['Intraday_High_Low_Spread'] = df['high'] - df['low']
    
    # Calculate Close-to-Open Return
    df['Close_to_Open_Return'] = (df['close'] - df['open']) / df['open']
    
    # Calculate True Range
    df['True_Range'] = df[['high' - df['low'], 
                           df['high'] - df['close'].shift(1), 
                           df['close'].shift(1) - df['low']]].max(axis=1)
    
    # Calculate Smoothed Price Momentum
    df['Daily_Price_Change'] = df['close'] - df['close'].shift(1)
    df['Smoothed_Price_Momentum'] = df['Daily_Price_Change'].rolling(window=10).mean()
    
    # Adjust Momentum by Volatility and Volume Trend
    range_ratio = df['High_Low_Range'] / df['close']
    df['Smoothed_Range_Ratio'] = range_ratio.rolling(window=20).mean()
    df['Smoothed_Volume_Trend'] = df['volume'].pct_change().rolling(window=20).mean()
    
    df['Adjusted_Momentum'] = df['Smoothed_Price_Momentum'] / df['Smoothed_Range_Ratio']
    df['Volume_Adjusted_Momentum'] = df['Adjusted_Momentum'] * df['Smoothed_Volume_Trend']
    
    # Final Alpha Factor
    df['Combined_Trend_Intraday'] = (df['Intraday_High_Low_Spread'] * df['Close_to_Open_Return']) / df['20_Day_Std_Log_Return']
    df['Final_Alpha_Factor'] = df['Combined_Trend_Intraday'] + df['Volume_Adjusted_Momentum'] + df['Factor']
    
    return df['Final_Alpha_Factor']
