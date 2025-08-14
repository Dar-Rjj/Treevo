import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Close-to-Open Return
    df['Close_to_Open_Return'] = df['open'].shift(-1) - df['close']
    
    # Volume Weighting
    df['Volume_Weighted_Return'] = df['Close_to_Open_Return'] * df['volume']
    
    # Determine Volatility using High, Low, and Close prices
    df['Daily_Range'] = df['high'] - df['low']
    initial_window = 20
    df['Rolling_Std'] = df['close'].rolling(window=initial_window).std()
    
    # Adjust Window Size based on Volatility
    def adjust_window(std):
        if std > df['Rolling_Std'].mean():
            return int(initial_window * 0.8)
        else:
            return int(initial_window * 1.2)
    
    adaptive_window = df['Rolling_Std'].apply(adjust_window)
    df['Adaptive_Window'] = adaptive_window
    
    # Enhanced Volatility Adjustments
    df['Price_Range'] = df['high'] - df['low']
    df['Percentage_Price_Range'] = (df['Price_Range'] / df['close']) * 100
    
    df['High_Low_Range'] = df['high'] - df['low']
    df['High_Prev_Close_Range'] = df['high'] - df['close'].shift(1)
    df['Prev_Close_Low_Range'] = df['close'].shift(1) - df['low']
    df['True_Range'] = df[['High_Low_Range', 'High_Prev_Close_Range', 'Prev_Close_Low_Range']].max(axis=1)
    
    df['Average_True_Range'] = df['True_Range'].rolling(window=initial_window).mean()
    
    # Adjust Weighting Factor based on Average True Range
    def adjust_weighting_factor(atr):
        if atr > df['Average_True_Range'].mean():
            return 0.8
        else:
            return 1.2
    
    weighting_factor = df['Average_True_Range'].apply(adjust_weighting_factor)
    df['Adjusted_Volume_Weighted_Return'] = df['Volume_Weighted_Return'] * weighting_factor
    
    # Rolling Statistics with Adaptive Window
    df['Rolling_Mean'] = df['Adjusted_Volume_Weighted_Return'].rolling(window=adaptive_window, min_periods=1).mean()
    df['Rolling_Std'] = df['Adjusted_Volume_Weighted_Return'].rolling(window=adaptive_window, min_periods=1).std()
    
    # Final Alpha Factor
    df['Alpha_Factor'] = df['Rolling_Mean'] / df['Rolling_Std']
    
    return df['Alpha_Factor']
