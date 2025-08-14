import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Price Movement
    df['Intraday_Range'] = df['High'] - df['Low']
    
    # Calculate Close to Open Difference
    df['CO_Difference'] = df['Close'] - df['Open']
    
    # Calculate Daily Log Return
    df['Daily_Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Calculate 20-Day Moving Average of Close Price
    df['20D_MA_Close'] = df['Close'].rolling(window=20).mean()
    
    # Calculate 20-Day Standard Deviation of Log Returns
    df['20D_STD_Log_Returns'] = df['Daily_Log_Return'].rolling(window=20).std()
    
    # Calculate 5-Day Intraday Volatility
    df['5D_Intraday_Volatility'] = df['Intraday_Range'].abs().rolling(window=5).sum()
    
    # Compute Intraday Stability
    df['Intraday_Stability'] = 1 / (df['5D_Intraday_Volatility'] / df['Intraday_Range'])
    
    # Calculate Trend Momentum Indicator
    df['Trend_Momentum_Indicator'] = ((df['Close'] - df['20D_MA_Close']) / df['Close']) * df['20D_STD_Log_Returns']
    
    # Calculate Volume Weighted Close-to-Open Return
    df['CO_Return'] = (df['Close'] - df['Open']) / df['Open']
    df['Volume_Weighted_CO_Return'] = df['CO_Return'] * df['Volume']
    
    # Volume Confirmation
    df['20D_Avg_Volume'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Confirmation'] = (df['Volume'] > df['20D_Avg_Volume']).astype(int)
    
    # Combine Factors
    df['Intraday_Momentum'] = df['Intraday_Range']
    df['Combined_Alpha'] = (
        df['Intraday_Momentum'] + 
        df['Volume_Weighted_CO_Return'] + 
        df['Intraday_Stability'] + 
        df['Trend_Momentum_Indicator'] + 
        df['Volume_Confirmation']
    )
    
    # Adjust for Long-Term Reversal
    df['Smoothed_Return'] = df['Close'].pct_change(periods=200).rolling(window=200).mean()
    df['Final_Alpha'] = df['Combined_Alpha'] - df['Smoothed_Return']
    
    return df['Final_Alpha']
