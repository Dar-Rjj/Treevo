import pandas as pd
import pandas as pd

def heuristics_v2(df):
    N = 5  # Lookback period for 5 days
    M = 10  # Lookback period for 10 days
    span = 7  # Span for EMA
    
    # Calculate Enhanced Price-Volume Momentum
    df['SMA_5'] = df['close'].rolling(window=N).mean()
    df['SMA_10'] = df['close'].rolling(window=M).mean()
    df['Price_Diff_5'] = (df['close'] - df['SMA_5']) / df['SMA_5']
    df['Price_Diff_10'] = (df['close'] - df['SMA_10']) / df['SMA_10']
    df['Cum_Volume_5'] = df['volume'].rolling(window=N).sum()
    df['Cum_Volume_10'] = df['volume'].rolling(window=M).sum()
    df['Momentum_Score_5'] = df['Price_Diff_5'] * df['Cum_Volume_5']
    df['Momentum_Score_10'] = df['Price_Diff_10'] * df['Cum_Volume_10']
    
    # Calculate Volume-Weighted Average Price (VWAP)
    df['HL_Mean'] = (df['high'] + df['low']) / 2
    df['VWAP'] = (df['HL_Mean'] * df['volume']).cumsum() / df['volume'].cumsum()
    
    # Calculate Daily Returns
    df['Daily_Return'] = df['close'].pct_change()
    
    # Calculate Volume-Weighted Momentum
    df['Daily_Return_Vol'] = df['Daily_Return'] * df['volume']
    df['Aggregate_Product'] = df['Daily_Return_Vol'].rolling(window=N).sum()
    df['Aggregate_Volume'] = df['volume'].rolling(window=N).sum()
    df['Volume_Weighted_Momentum'] = df['Aggregate_Product'] / df['Aggregate_Volume']
    
    # Smooth the Daily Return
    df['Smoothed_Return'] = df['Daily_Return'].ewm(span=span).mean()
    df['Volume_Adjusted_Daily_Return'] = df['Smoothed_Return'] * df['volume']
    
    # Calculate High-to-Low Price Range
    df['High_Low_Range'] = df['high'] - df['low']
    
    # Calculate Open-Adjusted Range
    df['Open_Adjusted_Range'] = df[['high' - df['open'], df['open'] - df['low']]].max(axis=1)
    
    # Calculate Trading Intensity
    df['Volume_Change'] = df['volume'].diff()
    df['Amount_Change'] = df['amount'].diff()
    df['Trading_Intensity'] = df['Volume_Change'] / df['Amount_Change']
    
    # Weight the Range by Trading Intensity
    df['Scaled_Trading_Intensity'] = df['Trading_Intensity'] / 1000
    df['Weighted_High_Low_Range'] = df['Scaled_Trading_Intensity'] * df['High_Low_Range']
    
    # Combine All Alpha Factors
    df['Alpha_Factor'] = (df['Momentum_Score_5'] + df['Momentum_Score_10'] 
                          + df['Volume_Adjusted_Daily_Return'] 
                          + df['Volume_Weighted_Momentum'] 
                          + df['Weighted_High_Low_Range'] 
                          + df['Open_Adjusted_Range'])
    
    return df['Alpha_Factor'].dropna()
