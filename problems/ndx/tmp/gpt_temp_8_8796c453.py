import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Volume-Weighted Price (VWP)
    df['VWP'] = (df['High'] * df['Volume'] + df['Low'] * df['Volume']) / (2 * df['Volume'])
    
    # Generate Intraday Volatility Metric
    df['Intraday_Volatility'] = (df['High'] - df['Low']) / df['Volume'].rolling(window=14).mean()
    
    # Rolling 14-Day High-to-Low Range Sum
    df['Rolling_14D_HL_Range_Sum'] = (df['High'] - df['Low']).rolling(window=14).sum()
    
    # Adjust by Intraday Volatility and 14-Day High-to-Low Range Sum
    df['HL_Range_Momentum'] = (df['VWP'] - df['VWP'].shift(1)) / (df['Intraday_Volatility'] * df['Rolling_14D_HL_Range_Sum'])
    
    # Calculate Volume-Adjusted Momentum
    df['VWP_20_EMA'] = df['VWP'].ewm(span=20, adjust=False).mean()
    df['Vol_Adjusted_Momentum'] = df['VWP'] - df['VWP_20_EMA']
    
    # Incorporate Recent Volatility
    df['Daily_Return'] = df['Close'].pct_change()
    df['Recent_Volatility'] = df['Daily_Return'].rolling(window=10).std()
    df['Vol_Adjusted_Momentum'] /= df['Recent_Volatility']
    
    # Combine Volume-Adjusted Momentum and Composite Volatility
    df['20_MA_High_Low_Range'] = (df['High'] - df['Low']).rolling(window=20).mean()
    df['20_MA_Open_Close_Range'] = (df['Open'] - df['Close']).rolling(window=20).mean()
    df['Composite_Volatility'] = df['20_MA_High_Low_Range'] + df['20_MA_Open_Close_Range']
    df['Combined_Momentum'] = df['Vol_Adjusted_Momentum'] - df['Composite_Volatility']
    
    # Calculate Daily Adjusted Return
    df['Daily_Adjusted_Return'] = (df['Close'] - df['Low'].shift(1)) / (df['High'].shift(1) - df['Low'].shift(1)) * 100
    
    # Apply Volume Smoothing
    df['7_Day_EWMA_Volume'] = df['Volume'].ewm(span=7, adjust=False).mean()
    df['Normalized_Volume'] = df['Volume'] / df['7_Day_EWMA_Volume']
    
    # Weighted Returns Calculation
    df['Weighted_Returns'] = df['Daily_Adjusted_Return'] * df['Normalized_Volume']
    df['Sum_Weighted_Returns_14D'] = df['Weighted_Returns'].rolling(window=14).sum()
    
    # Introduce Price Momentum Component
    df['14_Day_EWMA_Close'] = df['Close'].ewm(span=14, adjust=False).mean()
    df['Combined_Weighted_Returns'] = df['Sum_Weighted_Returns_14D'] + df['14_Day_EWMA_Close']
    
    # Separate Positive and Negative Weights
    df['Positive_Weighted_Returns'] = df['Combined_Weighted_Returns'].apply(lambda x: x if x > 0 else 0)
    df['Negative_Weighted_Returns'] = df['Combined_Weighted_Returns'].apply(lambda x: abs(x) if x < 0 else 0)
    
    df['Sum_Positive_Weighted_Returns_14D'] = df['Positive_Weighted_Returns'].rolling(window=14).sum()
    df['Sum_Negative_Weighted_Returns_14D'] = df['Negative_Weighted_Returns'].rolling(window=14).sum()
    
    # Compute Dynamic RSI
    df['Relative_Strength'] = df['Sum_Positive_Weighted_Returns_14D'] / df['Sum_Negative_Weighted_Returns_14D']
    df['Dynamic_RSI'] = 100 - (100 / (1 + df['Relative_Strength']))
    
    return df['Dynamic_RSI']
