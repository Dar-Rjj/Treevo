import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Simple Moving Average (SMA) of Close Prices Over 20 Days
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    
    # Exponential Moving Average (EMA) of Close Prices Over 12 Days
    df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
    
    # Crossover Signals
    df['Crossover_Signal'] = df['SMA_20'] - df['EMA_12']
    
    # Relative Strength Index (RSI) on Close Prices
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # Rate of Change (ROC) Indicator
    df['ROC_9'] = (df['close'] - df['close'].shift(9)) / df['close'].shift(9) * 100
    
    # On-Balance Volume (OBV)
    df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    
    # Volume Moving Average
    df['VMA_20'] = df['volume'].rolling(window=20).mean()
    
    # True Range
    df['True_Range'] = df[['high', 'low']].diff(axis=1).iloc[:, 1].abs().max(axis=1)
    
    # Donchian Channels
    df['Donchian_Upper_Band'] = df['high'].rolling(window=20).max()
    df['Donchian_Lower_Band'] = df['low'].rolling(window=20).min()
    
    # Historical Volatility
    df['HV_20'] = df['close'].pct_change().rolling(window=20).std() * np.sqrt(252)
    
    # Combined Factors
    df['Factor_1'] = df['SMA_20'] - df['EMA_12']
    df['Factor_2'] = df['RSI_14'] + df['ROC_9']
    df['Factor_3'] = df['OBV'] - df['VMA_20']
    df['Factor_4'] = df['True_Range'] / (df['high'] - df['low'])
    df['Factor_5'] = (df['close'] - df['Donchian_Lower_Band']) / (df['Donchian_Upper_Band'] - df['Donchian_Lower_Band'])
    
    # Final Alpha Factor Combinations
    df['Composite_Factor_1'] = (df['Factor_1'] + df['Factor_2']) * df['Factor_3']
    df['Composite_Factor_2'] = (df['Factor_4'] + df['Factor_5'])
    
    # Return the final composite factor
    return df['Composite_Factor_1'] + df['Composite_Factor_2']
