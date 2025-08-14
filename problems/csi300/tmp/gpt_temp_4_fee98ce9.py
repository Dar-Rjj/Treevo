import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Momentum
    df['Intraday_High_Low_Spread'] = df['High'] - df['Low']
    df['Prev_Open_to_Close_Return'] = df['Close'] - df['Open'].shift(1)
    
    # Calculate Volume Weighted Average Price (VWAP)
    prices = (df['High'] + df['Low'] + df['Close'] + df['Open']) / 4
    df['VWAP'] = (prices * df['Volume']).cumsum() / df['Volume'].cumsum()
    
    # Combine Intraday Momentum and VWAP
    df['Combined_Value'] = (df['VWAP'] - df['Intraday_High_Low_Spread']) * df['Volume']
    
    # Measure Relative Strength
    df['20_Day_SMA'] = df['Close'].rolling(window=20).mean()
    df['Relative_Strength'] = df['Close'] / df['20_Day_SMA']
    
    # Measure Volatility
    df['20_Day_STD'] = df['Close'].rolling(window=20).std()
    df['Normalized_Volatility'] = df['20_Day_STD'] / df['20_Day_SMA']
    
    # Measure Liquidity
    df['20_Day_Avg_Volume'] = df['Volume'].rolling(window=20).mean()
    df['Liquidity'] = df['Volume'] / df['20_Day_Avg_Volume']
    
    # Incorporate Relative Strength, Volatility, and Liquidity
    df['Adjusted_Combined_Value'] = df['Combined_Value'] * df['Relative_Strength'] / df['Normalized_Volatility'] * df['Liquidity']
    
    # Smooth the Factor with Adaptive Exponential Moving Average (EMA)
    alpha = 2 / (1 + df['20_Day_STD'].rolling(window=7).mean())
    df['Smoothed_Factor'] = df['Adjusted_Combined_Value'].ewm(alpha=alpha, adjust=False).mean()
    
    return df['Smoothed_Factor']
