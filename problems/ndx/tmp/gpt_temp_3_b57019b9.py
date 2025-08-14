import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Price-Volume Momentum
    N1 = 5
    N2 = 10
    df['EMA_5'] = df['close'].ewm(span=N1, adjust=False).mean()
    df['EMA_10'] = df['close'].ewm(span=N2, adjust=False).mean()
    
    df['Price_Diff_5'] = df['close'] - df['EMA_5']
    df['Price_Diff_10'] = df['close'] - df['EMA_10']
    
    df['Momentum_Score_5'] = df['Price_Diff_5'] / df['EMA_5']
    df['Momentum_Score_10'] = df['Price_Diff_10'] / df['EMA_10']
    
    df['Cum_Volume_5'] = df['volume'].rolling(window=N1).sum()
    df['Cum_Volume_10'] = df['volume'].rolling(window=N2).sum()
    
    df['Momentum_Adjusted_5'] = df['Momentum_Score_5'] * df['Cum_Volume_5']
    df['Momentum_Adjusted_10'] = df['Momentum_Score_10'] * df['Cum_Volume_10']

    # Calculate Volume-Weighted Average Price (VWAP)
    df['VWAP'] = (df['high'] + df['low']) / 2 * df['volume']
    df['Total_Volume'] = df['volume'].rolling(window=1).sum()
    df['VWAP'] = df['VWAP'].cumsum() / df['Total_Volume']

    # Calculate Daily Return using Close and VWAP
    df['Daily_Return'] = (df['close'] - df['VWAP']) / df['VWAP']

    # Smooth and Scale the Daily Return
    df['Smoothed_Return'] = df['Daily_Return'].ewm(span=5, adjust=False).mean()
    df['Volume_Adjusted_Return'] = df['Smoothed_Return'] * df['volume']

    # Calculate High-to-Low Price Range
    df['High_Low_Range'] = df['high'] - df['low']

    # Calculate Open-Adjusted Range
    df['Open_Adjusted_Range'] = df[['high', 'low']].apply(lambda x: max(x[0] - df['open'], df['open'] - x[1]), axis=1)

    # Calculate Trading Intensity
    df['Volume_Change'] = df['volume'].pct_change()
    df['Amount_Change'] = df['amount'].pct_change()
    df['Trading_Intensity'] = df['Volume_Change'] / df['Amount_Change']

    # Weight the Range by Trading Intensity
    scaling_factor = 100
    df['Weighted_High_Low_Range'] = df['High_Low_Range'] * (df['Trading_Intensity'] * scaling_factor)

    # Combine Momentum, Volume-Adjusted Daily Return, and Weighted Ranges
    df['Alpha_Factor'] = (
        df['Momentum_Adjusted_5'] + 
        df['Volume_Adjusted_Return'] + 
        df['Weighted_High_Low_Range'] + 
        df['Open_Adjusted_Range']
    )

    return df['Alpha_Factor']
