import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df, lookback=60):
    # Calculate High-Low Range Momentum
    df['Range'] = df['High'] - df['Low']
    df['Range_Momentum'] = (df['Range'] > df['Range'].shift(1)).astype(int) * 2 - 1

    # Calculate Close-to-Spread
    df['Close_to_Spread'] = np.maximum(df['Close'] - df['Low'], 0) + np.maximum(df['High'] - df['Close'], 0)

    # Calculate Volume-Weighted High-Low Range
    df['Volume_Weighted_Range'] = df['Range'] * df['Volume']

    # Calculate Volume-Adjusted Spread
    df['Volume_Adjusted_Spread'] = df['Close_to_Spread'].rolling(window=lookback).sum() / df['Volume'].rolling(window=lookback).sum()

    # Combine High-Low Range Momentum and Volume-Adjusted Spread
    df['Combined_Momentum_Volume'] = df['Range_Momentum'] * df['Volume_Adjusted_Spread']

    # Calculate Relative Strength
    df['Positive_Change'] = np.where(df['Close'] > df['Close'].shift(1), df['Close'] - df['Close'].shift(1), 0)
    df['Negative_Change'] = np.where(df['Close'] < df['Close'].shift(1), df['Close'].shift(1) - df['Close'], 0)
    df['Relative_Strength'] = df['Positive_Change'].rolling(window=lookback).sum() / df['Negative_Change'].rolling(window=lookback).sum()
    df['RS_Factor'] = df['Relative_Strength'].ewm(span=lookback, adjust=False).mean()

    # Integrate Intraday Factors
    df['Intraday_High_Low_Difference'] = df['High'] - df['Low']
    df['Intraday_High_Low_Ratio'] = (df['High'] - df['Low']) / df['Low']
    df['Intraday_Factor'] = (df['Intraday_High_Low_Difference'] * df['Volume_Weighted_Range']) + (df['Intraday_High_Low_Ratio'] * (df['Close'] - df['Open']) / df['Open'])

    # Integrate VWAP into Alpha Factor
    df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    df['VWAP_Momentum'] = df['VWAP'].rolling(window=lookback).sum()

    # Combine All Integrated Factors
    df['Integrated_Factor'] = (df['Combined_Momentum_Volume'] * df['RS_Factor']) + df['Intraday_Factor'] + df['VWAP_Momentum']

    # Adjust for Volatility and Smooth Over Time
    df['Daily_Returns'] = df['Close'].pct_change()
    df['Daily_Volatility'] = df['Daily_Returns'].rolling(window=lookback).std()
    df['Smoothed_Factor'] = df['Integrated_Factor'].ewm(span=lookback, adjust=False).mean() / df['Daily_Volatility']

    return df['Smoothed_Factor'].dropna()
