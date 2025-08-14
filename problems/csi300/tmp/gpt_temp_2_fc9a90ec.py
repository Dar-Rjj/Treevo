import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Momentum
    df['High_Low_Range'] = df['high'] - df['low']
    df['Intraday_Momentum'] = (df['High_Low_Range'] / df['open']) * 100
    
    # Calculate Enhanced Momentum Based on Close Prices
    df['Delta_Close'] = df['close'].diff()
    
    # Incorporate Volume into Momentum
    C = 200
    df['Scaled_Momentum'] = df['Delta_Close'] / (df['volume'] ** (1/3)) * C
    
    # Adjust Scaled Momentum by Intraday High-Low Spread
    df['Daily_Price_Range'] = df['high'] - df['low']
    df['Adjusted_Scaled_Momentum'] = df['Scaled_Momentum'] / (df['Daily_Price_Range'] ** 2)
    
    # Calculate Volume Flow
    df['Volume_Diff'] = df['volume'].diff()
    df['Average_Volume'] = (df['volume'] + df['volume'].shift(1)) / 2
    df['Volume_Flow'] = df['Volume_Diff'] / df['Average_Volume']
    
    # Combine Momentum and Volume Flow
    df['Combined_Intraday_Momentum'] = df['Intraday_Momentum'] * df['Volume_Flow']
    df['Combined_Adjusted_Scaled_Momentum'] = df['Adjusted_Scaled_Momentum'] * df['Volume_Flow']
    
    # Compute Intraday Volatility
    df['Intraday_Volatility'] = df[['open', 'high', 'low', 'close']].mad(axis=1)
    
    # Weight by Intraday Volatility
    df['Weighted_Combined_Factor'] = df['Combined_Intraday_Momentum'] * df['Intraday_Volatility']
    
    # Calculate Short-Term Moving Average (7 days) and Long-Term Moving Average (30 days)
    df['Short_Term_MA'] = df['close'].rolling(window=7).mean()
    df['Long_Term_MA'] = df['close'].rolling(window=30).mean()
    
    # Subtract Short-Term from Long-Term MA
    df['MA_Crossover'] = df['Short_Term_MA'] - df['Long_Term_MA']
    
    # Calculate Intraday Momentum Reversal Indicator
    df['Intraday_Return'] = (df['high'] - df['low']) / df['low']
    df['Lagged_Intraday_Return'] = df['Intraday_Return'].shift(1)
    df['Reversal_Indicator'] = df['Intraday_Return'] - df['Lagged_Intraday_Return']
    
    # Summarize Momentum Over Multiple Days
    N = 5
    df['Cumulative_Product'] = (1 + df['Weighted_Combined_Factor']).rolling(window=N).apply(np.prod, raw=True) - 1
    
    # Final Momentum Indicator
    df['Final_Combined_Momentum'] = np.sqrt(df['Combined_Adjusted_Scaled_Momentum'] * df['Cumulative_Product'])
    
    # Combine All Indicators
    df['Combined_Result'] = (df['MA_Crossover'] + 
                             df['Reversal_Indicator'] + 
                             df['Weighted_Combined_Factor'] + 
                             df['Final_Combined_Momentum'])
    
    # Determine Final Signal
    df['Final_Signal'] = np.where(df['Combined_Result'] > 0, 1, -1)
    
    # Assign Weight by Volume
    df['Factor'] = df['Final_Signal'] * df['volume']
    
    return df['Factor'].dropna()
