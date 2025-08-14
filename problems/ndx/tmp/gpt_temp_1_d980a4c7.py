import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df, N=10, ema_span=5):
    # Calculate Volume-Weighted Average Price (VWAP)
    df['TypicalPrice'] = (df['High'] + df['Low']) / 2
    df['TP_Volume'] = df['TypicalPrice'] * df['Volume']
    df['Cumulative_TP_Volume'] = df['TP_Volume'].rolling(window=N).sum()
    df['Cumulative_Volume'] = df['Volume'].rolling(window=N).sum()
    df['VWAP'] = df['Cumulative_TP_Volume'] / df['Cumulative_Volume']
    
    # Calculate Daily Returns
    df['Daily_Return'] = df['Close'].pct_change()
    
    # Calculate Volume-Weighted Momentum
    df['Return_Volume'] = df['Daily_Return'] * df['Volume']
    df['Cumulative_Return_Volume'] = df['Return_Volume'].rolling(window=N).sum()
    df['VWAM'] = df['Cumulative_Return_Volume'] / df['Cumulative_Volume']
    
    # Calculate VWAP Adjusted Daily Return
    df['VWAP_Adjusted_Return'] = (df['Close'] - df['VWAP']) / df['VWAP']
    
    # Smooth and Scale the VWAP Adjusted Daily Return
    df['Smoothed_VWAP_Adjusted_Return'] = df['VWAP_Adjusted_Return'].ewm(span=ema_span).mean()
    df['Scaled_Smoothed_VWAP_Adjusted_Return'] = df['Smoothed_VWAP_Adjusted_Return'] * df['Volume']
    
    # Calculate High-to-Low Range
    df['Range'] = df['High'] - df['Low']
    
    # Calculate Open-Adjusted Range
    df['Open_Adjusted_Range'] = np.maximum(df['High'] - df['Open'], df['Open'] - df['Low'])
    
    # Combine Momentum, VWAP, and Range
    df['Combined_Factor'] = df['VWAM'] + df['Scaled_Smoothed_VWAP_Adjusted_Return'] + df['Open_Adjusted_Range']
    
    # Calculate Daily Range Factor
    df['Daily_Range_Factor_1'] = (df['High'] - df['Low']) / df['Open']
    df['Daily_Range_Factor_2'] = (df['High'] - df['Open']) / df['Close']
    df['Daily_Range_Factor_3'] = (df['Open'] - df['Low']) / df['Close']
    
    # Additional Features
    df['Close_Diff'] = df['Close'] - df['Close'].shift(1)
    df['Average_Open_Close'] = (df['Open'] + df['Close']) / 2
    df['Volume_Ratio'] = df['Volume'] / df['Average_Open_Close']
    
    # Final Combined Alpha Factor
    df['Alpha_Factor'] = df['Combined_Factor'] + df['Daily_Range_Factor_1'] + df['Daily_Range_Factor_2'] + df['Daily_Range_Factor_3'] + df['Close_Diff'] + df['Volume_Ratio']
    
    return df['Alpha_Factor'].dropna()
