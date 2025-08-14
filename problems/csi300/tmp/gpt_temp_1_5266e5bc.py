import pandas as pd
import numpy as np
import pandas as pd

def heuristics_v2(df):
    # Calculate Daily Price Movement
    df['Daily_Price_Movement'] = df['close'] - df['close'].shift(1)
    
    # Calculate Price Change Over Time Window (20 days)
    df['Price_Change_20D'] = df['close'] - df['close'].shift(20)
    
    # Calculate Historical Price Volatility (22 days)
    df['Volatility'] = df['close'].rolling(window=22).std()
    
    # Calculate Momentum (7 and 25 days)
    df['Momentum_7D'] = df['close'] / df['close'].shift(7) - 1
    df['Momentum_25D'] = df['close'] / df['close'].shift(25) - 1
    
    # Incorporate Daily Volume Changes
    df['Volume_Change'] = (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1)
    
    # Aggregate Volume Impact
    df['Volume_Impact'] = (df['Volume_Change'] * df['close']).rolling(window=22).sum() / df['close'].rolling(window=22).sum()
    
    # Calculate Volume Direction
    df['Volume_Direction'] = df['volume'].apply(lambda x: 1 if x > df['volume'].shift(1) else -1)
    
    # Combine Price Movement and Volume Direction
    df['Price_Volume_Combined'] = df['Daily_Price_Movement'] * df['Volume_Direction']
    
    # Weight by Volume and Inverse Volatility
    df['Inverse_Volatility'] = 1 / df['Volatility']
    df['Weighted_Price_Volume'] = df['Price_Volume_Combined'] * df['volume'] * df['Inverse_Volatility']
    
    # Raw Daily Return
    df['Raw_Return'] = df['close'] / df['close'].shift(1) - 1
    
    # Rolling Sum Returns (e.g., 10-day window)
    df['Rolling_Sum_Returns'] = df['Raw_Return'].rolling(window=10).sum()
    
    # Weighted Price Movement
    df['Average_Volume'] = df['volume'].rolling(window=10).mean()
    df['Weighted_Price_Movement'] = df['Rolling_Sum_Returns'] * df['Average_Volume']
    
    # Lagged Momentum Value
    df['Lagged_Momentum'] = df['Rolling_Sum_Returns'].shift(10)
    
    # Subtract Lagged Momentum
    df['Adjusted_Price_Movement'] = df['Weighted_Price_Movement'] - df['Lagged_Momentum']
    
    # Combine Weights
    df['Combined_Weights'] = df['Volume_Impact'] * df['Inverse_Volatility'] * df['Price_Change_20D']
    
    # Final Adjustment
    df['Final_Factor'] = df['Combined_Weights'] + df['Adjusted_Price_Movement']
    
    # 5-day Simple Moving Average (SMA) of Close Prices
    df['SMA_5D'] = df['close'].rolling(window=5).mean()
    df['SMA_5D_Diff'] = df['close'] - df['SMA_5D']
    
    # Volatility of Daily Returns (20 days)
    df['Log_Returns'] = df['close'].pct_change().apply(lambda x: np.log(1 + x))
    df['Volatility_Returns'] = df['Log_Returns'].rolling(window=20).std()
    
    # High-Low Range Indicator
    df['High_Low_Range'] = df['high'] - df['low']
    df['Avg_High_Low_Range'] = df['High_Low_Range'].rolling(window=20).mean()
    
    # Normalized High-Low Range
    df['Normalized_High_Low_Range'] = df['High_Low_Range'] / df['Avg_High_Low_Range']
    
    # Adjust Final Factor
    df['Final_Factor'] = df['Final_Factor'] * df['Normalized_High_Low_Range']
    
    # Output Final Alpha Factor
    return df['Final_Factor']
