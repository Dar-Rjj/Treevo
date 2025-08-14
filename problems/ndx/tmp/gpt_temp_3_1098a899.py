import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate High-to-Low Range
    df['Range'] = df['High'] - df['Low']
    
    # Compute High-Low Ratio and Intraday Trading Activity
    df['IntradayMomentum'] = (df['High'] - df['Low']) / df['Low'] * df['Volume'] / (df['High'] + df['Low'])
    
    # Measure Sustained 5-Day Momentum
    df['Sustained5DayMomentum'] = (df['Close'].diff() / df['Close'].shift()).rolling(window=5).mean()
    
    # Combine Intraday and 5-Day Momentum
    df['CombinedMomentum'] = df['IntradayMomentum'] * df['Sustained5DayMomentum'] * df['Volume']
    
    # Determine Reversal Signal
    df['5DayEMA'] = df['Close'].ewm(span=5, adjust=False).mean()
    df['10DayEMA'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['ReversalSignal'] = np.where(df['5DayEMA'] > df['10DayEMA'], 1, -1)
    df['RollingMedianRange'] = df['Range'].rolling(window=14).median()
    df['AdjustedReversalSignal'] = df['ReversalSignal'] * (1 + (df['RollingMedianRange'] / df['Range']))
    
    # Filter by Volume
    volume_threshold = df['Volume'].quantile(0.75)
    df['FilteredSignal'] = np.where(df['Volume'] > volume_threshold, df['AdjustedReversalSignal'], 0)
    
    # Incorporate Volatility into Alpha Factor
    df['HistoricalVolatility'] = df['Close'].rolling(window=21).std()
    df['VolatilityAdjustedFactor'] = df['CombinedMomentum'] / df['HistoricalVolatility']
    
    # Enhance Momentum with Volume-Weighted Moving Averages
    df['5DayVWMA'] = (df['Close'] * df['Volume']).rolling(window=5).sum() / df['Volume'].rolling(window=5).sum()
    df['10DayVWMA'] = (df['Close'] * df['Volume']).rolling(window=10).sum() / df['Volume'].rolling(window=10).sum()
    df['5DaySMA'] = df['Close'].rolling(window=5).mean()
    df['10DaySMA'] = df['Close'].rolling(window=10).mean()
    df['VWMAMomentumAdjustment'] = np.where(df['5DayVWMA'] > df['5DaySMA'], 1, -1) * np.where(df['10DayVWMA'] < df['10DaySMA'], -1, 1)
    
    # Final Alpha Factor
    df['FinalAlphaFactor'] = df['VolatilityAdjustedFactor'] * df['FilteredSignal'] * df['VWMAMomentumAdjustment']
    
    return df['FinalAlphaFactor']
