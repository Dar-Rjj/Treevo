import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Volume Weighted Average Price (VWAP)
    df['VWAP'] = (df[['High', 'Low', 'Close']].mean(axis=1) * df['Volume']) / df['Volume']

    # Calculate Volume Increase Ratio
    df['Volume_t-1'] = df['Volume'].shift(1)
    df['Volume_Increase_Ratio'] = df['Volume'] / df['Volume_t-1']
    
    # Calculate Close Price Momentum
    df['Avg_Close_t-5_to_t-1'] = df['Close'].shift(1).rolling(window=5).mean()
    df['Close_Price_Momentum'] = df['Close'] - df['Avg_Close_t-5_to_t-1']
    
    # Calculate High-Low Range Ratio
    df['High_Low_Range'] = df['High'] - df['Low']
    
    # Calculate Open-Close Spread
    df['Open_Close_Spread'] = df['Close'] - df['Open']
    
    # Combine Factors
    df['Interim_Factor_1'] = (df['Volume_Increase_Ratio'] * df['Close_Price_Momentum']) / df['High_Low_Range']
    df['Interim_Factor_2'] = df['Interim_Factor_1'] * df['Open_Close_Spread']
    df['Final_Factor'] = df['Interim_Factor_2'] * df['VWAP']
    
    return df['Final_Factor'].dropna()
