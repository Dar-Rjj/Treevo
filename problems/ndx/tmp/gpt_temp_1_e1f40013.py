import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Volume-Weighted Price (VWP)
    df['VWP'] = (df['High'] * df['Volume'] + df['Low'] * df['Volume']) / (2 * df['Volume'])
    
    # Assess Intraday Volatility
    df['Intraday_Range'] = df['High'] - df['Low']
    df['Average_Volume_14D'] = df['Volume'].rolling(window=14).mean()
    df['Intraday_Volatility_Adjusted'] = df['Intraday_Range'] / df['Average_Volume_14D']
    df['Intraday_Volatility_Metric'] = df['Intraday_Range'] * df['Intraday_Volatility_Adjusted']
    
    # Integrate High-to-Low Range Momentum
    df['HL_Range_14D_Sum'] = (df['High'] - df['Low']).rolling(window=14).sum()
    df['VWP_Diff'] = df['VWP'].diff()
    df['Momentum_Adjusted'] = df['VWP_Diff'] / (df['Intraday_Volatility_Metric'] + df['HL_Range_14D_Sum'])
    
    # Calculate Volume-Adjusted Momentum
    df['VWP_20D_EMA'] = df['VWP'].ewm(span=20, adjust=False).mean()
    df['VWP_Momentum'] = df['VWP'] - df['VWP_20D_EMA']
    
    # Incorporate Recent Volatility
    df['Daily_Returns'] = df['Close'].pct_change()
    df['Recent_Volatility'] = df['Daily_Returns'].rolling(window=10).std()
    df['Vol_Adjusted_Momentum'] = df['VWP_Momentum'] / (df['Recent_Volatility'] + 1e-6)  # add small constant to avoid division by zero
    
    # Calculate High-Low Range Volatility
    df['High_Low_Range'] = df['High'] - df['Low']
    df['High_Low_Range_20D_MA'] = df['High_Low_Range'].rolling(window=20).mean()
    
    # Calculate Open-Close Range Volatility
    df['Open_Close_Range'] = df['Open'] - df['Close']
    df['Open_Close_Range_20D_MA'] = df['Open_Close_Range'].rolling(window=20).mean()
    
    # Combine Volume-Adjusted Momentum and Composite Volatility
    df['Composite_Volatility'] = (df['High_Low_Range_20D_MA'] + df['Open_Close_Range_20D_MA']) / 2
    df['Alpha_Factor'] = df['Vol_Adjusted_Momentum'] - df['Composite_Volatility']
    
    return df['Alpha_Factor']
