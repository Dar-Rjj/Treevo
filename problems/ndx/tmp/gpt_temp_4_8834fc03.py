import pandas as pd
import pandas as pd

def heuristics_v2(df, n=20, m=14):
    # Calculate Daily Return
    df['Daily_Return'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Calculate Volume Change Ratio
    df['Volume_Change_Ratio'] = df['volume'] / df['volume'].shift(1)
    
    # Compute Weighted Momentum
    df['Weighted_Momentum'] = (df['Daily_Return'] * df['Volume_Change_Ratio']).rolling(window=n).sum()
    
    # Adjust for Price Volatility
    df['True_Range'] = df[['high' - 'low', abs('high' - 'close').shift(1), abs('low' - 'close').shift(1)]].max(axis=1)
    df['ATR'] = df['True_Range'].rolling(window=m).mean()
    df['ATR_Adjusted_Momentum'] = df['Weighted_Momentum'] - df['ATR']
    
    # Calculate 5-Day Simple Moving Average (SMA) of Close price
    df['SMA_5'] = df['close'].rolling(window=5).mean()
    
    # Compute Price Difference
    df['Price_Difference'] = df['close'] - df['SMA_5']
    
    # Compute Momentum Score
    df['Momentum_Score'] = df['Price_Difference'] / df['SMA_5']
    
    # Calculate Cumulative Volume
    df['Cumulative_Volume'] = df['volume'].rolling(window=5).sum()
    
    # Adjust Momentum Score by Cumulative Volume
    df['Adjusted_Momentum_Score'] = df['Momentum_Score'] * df['Cumulative_Volume']
    
    # Calculate High-to-Low Price Range
    df['High_Low_Range'] = df['high'] - df['low']
    
    # Calculate Trading Intensity
    df['Volume_Change'] = df['volume'] - df['volume'].shift(1)
    df['Amount_Change'] = df['amount'] - df['amount'].shift(1)
    df['Trading_Intensity'] = df['Volume_Change'] / df['Amount_Change']
    
    # Weight the Range by Trading Intensity
    df['Weighted_Range'] = (df['Trading_Intensity'] * 1000) * df['High_Low_Range']
    
    # Combine Adjusted Momentum, Weighted Range, and ATR-Adjusted Momentum
    df['Combined_Factor'] = df['Adjusted_Momentum_Score'] + df['Weighted_Range'] + df['ATR_Adjusted_Momentum']
    
    # Calculate 5-Day Exponential Moving Average (EMA) of Close price
    df['EMA_5'] = df['close'].ewm(span=5, adjust=False).mean()
    
    # Compute EMA-Based Momentum
    df['EMA_Based_Momentum'] = df['close'] - df['EMA_5']
    
    # Incorporate EMA-Based Momentum
    df['Final_Factor'] = df['Combined_Factor'] + df['EMA_Based_Momentum']
    
    return df['Final_Factor']
