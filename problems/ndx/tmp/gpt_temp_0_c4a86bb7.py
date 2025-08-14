import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate 20-day Price Momentum
    df['Momentum'] = df['Close'] - df['Close'].shift(20)
    
    # Identify Breakout Days
    df['High_Low_Range'] = df['High'] - df['Low']
    avg_high_low_range = df['High_Low_Range'].rolling(window=20).mean()
    df['Breakout'] = (df['High_Low_Range'] > 2 * avg_high_low_range).astype(int)
    
    # Calculate Volume-Adjusted Breakout Impact
    df['Daily_Return'] = (df['Close'] - df['Open']) / df['Open']
    df['Volume_Adjusted_Return'] = df['Daily_Return'] * df['Volume']
    df['Volume_Adjusted_Breakout_Impact'] = df[df['Breakout'] == 1]['Volume_Adjusted_Return'].rolling(window=20).sum().fillna(0)
    
    # Integrate Volume Trend Impact
    df['Volume_Change'] = df['Volume'] - df['Volume'].shift(1)
    df['Volume_Growth_Rate'] = df['Volume_Change'].ewm(span=5, adjust=False).mean()
    
    # Compute Volume and Amount Volatility
    df['Volume_Volatility'] = df['Volume'].diff().rolling(window=20).std().fillna(0)
    df['Amount_Volatility'] = df['Amount'].diff().rolling(window=20).std().fillna(0)
    df['Combined_Volatility'] = df['Volume_Volatility'] + df['Amount_Volatility']
    
    # Adjust Momentum by Combined Volatility
    df['Adjusted_Momentum'] = df['Momentum'] / (np.sqrt(df['Combined_Volatility']) + 1e-6)
    
    # Calculate Intraday Price Range
    df['Intraday_Price_Range'] = df['High'] - df['Low']
    
    # Calculate Close to Open Difference
    df['Close_Open_Diff'] = df['Close'] - df['Open']
    
    # Calculate Average Transaction Size
    df['Avg_Transaction_Size'] = df['Amount'] / df['Volume']
    
    # Calculate Intraday Volatility
    df['Intraday_Volatility'] = (df['High'] - df['Low']) / df['Close']
    
    # Calculate Transaction Size Variance
    df['Transaction_Size_Variance'] = (df['Amount'] / df['Volume'] - df['Avg_Transaction_Size']) ** 2
    
    # Integrate Adjusted Momentum, Volume-Adjusted Breakout Impact, and Close-Open Spread
    df['Integrated_Result'] = (df['Adjusted_Momentum'] + df['Volume_Adjusted_Breakout_Impact'] + 
                              df['Close_Open_Diff'] * np.sign(df['Close_Open_Diff']))
    
    # Combine Factors
    df['Alpha_Factor'] = (
        (df['Close_Open_Diff'] * df['Close_Open_Diff'] * df['Volume'] + df['Avg_Transaction_Size']) * 
        (df['Intraday_Volatility'] / df['Intraday_Price_Range']) * 
        (1 + df['Transaction_Size_Variance']) * 
        (df['Volume_Adjusted_Breakout_Impact'] + df['Adjusted_Momentum']) * 
        df['Volume_Growth_Rate']
    )
    
    return df['Alpha_Factor']
