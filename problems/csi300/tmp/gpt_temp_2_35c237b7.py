import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Price Range
    df['Price_Range'] = df['high'] - df['low']
    
    # Compute Intraday Return
    df['Intraday_Return'] = (df['close'] - df['open']) / df['open']
    
    # Determine Intraday Volatility (True Range)
    df['Prev_Close'] = df['close'].shift(1)
    df['True_Range'] = df[['high', 'low']].sub(df['Prev_Close'], axis=0).abs().max(axis=1)
    df['True_Range'] = df[['high' - df['low'], df['high'] - df['Prev_Close'], df['low'] - df['Prev_Close']]].max(axis=1)
    
    # Adjust Volatility by Volume
    df['Adjusted_Volatility'] = df['True_Range'] * df['volume']
    
    # Calculate Intraday Momentum
    df['Open_t-1'] = df['open'].shift(1)
    df['Intraday_Momentum'] = df['close'] - df['Open_t-1']
    
    # Combine Intraday Return, Adjusted Volatility, and Momentum
    df['Combined_Value'] = (df['Intraday_Return'] 
                            - df['Adjusted_Volatility'] 
                            + df['Intraday_Momentum'])
    
    # Generate Final Alpha Factor
    df['Alpha_Factor'] = (df['Combined_Value'] > 0).astype(int)
    
    return df['Alpha_Factor']
