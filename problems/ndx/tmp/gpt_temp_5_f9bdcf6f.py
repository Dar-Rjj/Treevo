import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Volume-Adjusted Price (VAP)
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['VAP'] = df['Typical_Price'] * df['Volume']

    # Compute Daily Log Returns
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))

    # Calculate Volume-Adjusted Momentum
    df['VAP_20_EMA'] = df['VAP'].ewm(span=20, adjust=False).mean()
    df['Momentum'] = df['VAP'] - df['VAP_20_EMA']

    # Incorporate Recent Volatility
    df['Volatility'] = df['Log_Returns'].rolling(window=10).std()
    df['Momentum_Adj_Vol'] = df['Momentum'] / df['Volatility']

    # Calculate High-Low Range Volatility
    df['HL_Range'] = df['High'] - df['Low']
    df['HL_Volatility'] = df['HL_Range'].rolling(window=20).std()

    # Calculate Open-Close Range Volatility
    df['OC_Range'] = df['Open'] - df['Close']
    df['OC_Volatility'] = df['OC_Range'].rolling(window=20).std()

    # Combine Volume-Adjusted Momentum and Composite Volatility
    df['Composite_Volatility'] = (df['HL_Volatility'] + df['OC_Volatility']) / 2
    df['Momentum_Composite_Vol'] = df['Momentum'] - df['Composite_Volatility']

    # Calculate Rolling Average of High-to-Low Range
    df['Rolling_HL_Avg'] = df['HL_Range'].rolling(window=20).mean()

    # Calculate Daily Price Change
    df['Price_Change'] = df['Close'] - df['Open']

    # Compute 5-Day Moving Average of Volume-Weighted Price Change
    df['Vol_Weighted_Price_Change'] = df['Price_Change'] * df['Volume']
    df['5_Day_Vol_Weighted_MA'] = df['Vol_Weighted_Price_Change'].rolling(window=5).sum() / df['Volume'].rolling(window=5).sum()

    # Compute 10-Day Moving Average of Price Change
    df['10_Day_Price_Change_MA'] = df['Price_Change'].rolling(window=10).mean()

    # Determine Reversal Signal
    df['Reversal_Signal'] = np.where(df['5_Day_Vol_Weighted_MA'] > df['10_Day_Price_Change_MA'], 1, 
                                     np.where(df['5_Day_Vol_Weighted_MA'] < df['10_Day_Price_Change_MA'], -1, 0))
    df['Reversal_Signal'] *= df['Rolling_HL_Avg']

    # Analyze Price Reversals Using High, Low, and Open Prices
    df['Daily_Price_Reversal'] = (df['High'] - df['Low']) / df['Open']
    df['20_Day_Price_Reversal'] = df['Daily_Price_Reversal'].rolling(window=20).sum()

    # Assess Volume Impact on Price Movement
    df['Volume_Weighted_Close'] = df['Close'] * df['Volume']
    df['20_Day_Volume_Weighted_EMA'] = df['Volume_Weighted_Close'].ewm(span=20, adjust=False).mean()

    # Filter by Volume
    volume_threshold = df['Volume'].quantile(0.75)
    df['Filtered_Signal'] = np.where(df['Volume'] > volume_threshold, 1, 0)

    # Combine Factors for Final Alpha
    df['Final_Alpha'] = (0.4 * df['Momentum_Composite_Vol'] + 
                         0.3 * df['Reversal_Signal'] + 
                         0.2 * df['20_Day_Price_Reversal'] + 
                         0.1 * df['20_Day_Volume_Weighted_EMA']) * df['Filtered_Signal']

    return df['Final_Alpha'].dropna()
