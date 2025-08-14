import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df, N=14):
    # Calculate Daily Price Change
    df['Price_Change'] = df['close'].diff()
    
    # Compute Volume Adjusted Return
    df['Volume_Adjusted_Return'] = df['Price_Change'] / df['volume']
    
    # Initialize EMA of Volume Adjusted Returns
    df['EMA_Volume_Adjusted_Return'] = 0
    ema_var = df['Volume_Adjusted_Return'].iloc[0]
    multiplier_var = 2 / (N + 1)
    
    # Calculate EMA of Volume Adjusted Returns
    for i in range(1, len(df)):
        ema_var = (df['Volume_Adjusted_Return'].iloc[i] * multiplier_var) + (ema_var * (1 - multiplier_var))
        df.loc[df.index[i], 'EMA_Volume_Adjusted_Return'] = ema_var
    
    # Calculate High-Low Range
    df['High_Low_Range'] = df['high'] - df['low']
    
    # Calculate Volume Weighted High-Low Range
    df['Volume_Weighted_High_Low_Range'] = df['High_Low_Range'] * df['volume']
    
    # Initialize EMA of Volume Weighted High-Low Range
    df['EMA_Volume_Weighted_High_Low_Range'] = 0
    ema_vwhlr = df['Volume_Weighted_High_Low_Range'].iloc[0]
    multiplier_vwhlr = 2 / (N + 1)
    
    # Calculate EMA of Volume Weighted High-Low Range
    for i in range(1, len(df)):
        ema_vwhlr = (df['Volume_Weighted_High_Low_Range'].iloc[i] * multiplier_vwhlr) + (ema_vwhlr * (1 - multiplier_vwhlr))
        df.loc[df.index[i], 'EMA_Volume_Weighted_High_Low_Range'] = ema_vwhlr
    
    # Generate Final Alpha Factor
    df['Alpha_Factor'] = df['EMA_Volume_Adjusted_Return'] - df['EMA_Volume_Weighted_High_Low_Range']
    
    return df['Alpha_Factor']
