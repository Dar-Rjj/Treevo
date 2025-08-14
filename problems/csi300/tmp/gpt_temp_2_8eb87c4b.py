import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Close-to-Open Return
    df['Close_to_Open_Return'] = df['open'].shift(-1) - df['close']
    
    # Volume Weighting
    df['Volume_Weighted_Return'] = df['Volume_Weighted_Return'] * df['volume']
    
    # Determine Volatility using High, Low, and Close prices
    df['Volatility'] = df[['high', 'low', 'close']].rolling(window=20).std().mean(axis=1)
    
    # Define a function to adjust the window size based on volatility
    def get_adaptive_window(volatility, high_vol_threshold=0.05, low_vol_threshold=0.01, min_window=5, max_window=60):
        if volatility > high_vol_threshold:
            return min_window
        elif volatility < low_vol_threshold:
            return max_window
        else:
            return (max_window - min_window) * (high_vol_threshold - volatility) / (high_vol_threshold - low_vol_threshold) + min_window
    
    # Apply the adaptive window size
    df['Adaptive_Window'] = df['Volatility'].apply(get_adaptive_window)
    
    # Calculate Rolling Mean of Volume Weighted Close-to-Open Return with Adaptive Window
    df['Rolling_Mean'] = df['Volume_Weighted_Return'].rolling(window=df['Adaptive_Window'], min_periods=1).mean()
    
    # Calculate Rolling Standard Deviation of Volume Weighted Close-to-Open Return with Adaptive Window
    df['Rolling_Std'] = df['Volume_Weighted_Return'].rolling(window=df['Adaptive_Window'], min_periods=1).std()
    
    # Normalize the factor by subtracting the rolling mean and dividing by the rolling standard deviation
    df['Factor_Value'] = (df['Volume_Weighted_Return'] - df['Rolling_Mean']) / df['Rolling_Std']
    
    # Return the final factor value
    return df['Factor_Value']
