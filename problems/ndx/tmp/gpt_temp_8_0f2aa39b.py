import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Volume-Weighted Average Price (VWAP)
    df['VWAP'] = (df['High'] + df['Low']) / 2 * df['Volume']
    df['VWAP'] = df['VWAP'].cumsum() / df['Volume'].cumsum()
    
    # Calculate Daily Return using Close price and VWAP
    df['Daily_Return'] = (df['Close'] - df['VWAP']) / df['VWAP']
    
    # Smooth the Daily Return using Exponential Moving Average (EMA)
    span = 5  # Set the span for EMA
    df['Smoothed_Return'] = df['Daily_Return'].ewm(span=span, adjust=False).mean()
    
    # Multiply Smoothed Return by Volume to get the initial alpha factor
    df['Alpha_Factor'] = df['Smoothed_Return'] * df['Volume']
    
    # Calculate High-to-Low Return
    df['High_Low_Return'] = (df['High'] - df['Low']) / df['Low']
    
    # Weight by Volume
    df['Weighted_High_Low_Return'] = df['High_Low_Return'] * df['Volume']
    
    # Detect Volume Spikes
    df['Volume_Change'] = df['Volume'] / df['Volume'].shift(1)
    spike_threshold = 1.5  # Define the spike threshold
    df['Volume_Spike'] = df['Volume_Change'] > spike_threshold
    
    # Combine Weighted Return and Spike Detection
    df['Enhanced_Alpha_Factor'] = df['Weighted_High_Low_Return']
    df.loc[df['Volume_Spike'], 'Enhanced_Alpha_Factor'] = df['Weighted_High_Low_Return'] * 2  # Increase factor value if spike detected
    
    return df['Enhanced_Alpha_Factor']
