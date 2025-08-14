import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Daily Price Change
    df['Daily_Price_Change'] = df['close'] - df['close'].shift(1)
    
    # Compute 10-Day EMA of Daily Price Changes
    df['10Day_EMA_Price_Change'] = df['Daily_Price_Change'].ewm(span=10, min_periods=10).mean()
    
    # Calculate Intraday Return
    df['Intraday_Return'] = (df['high'] - df['low']) / df['low']
    
    # Calculate Close-to-Open Return
    df['Close_to_Open_Return'] = (df['close'] - df['open']) / df['open']
    
    # Combine Intraday and Close-to-Open Returns
    intraday_weight = np.abs(df['Intraday_Return'])
    close_to_open_weight = np.abs(df['Close_to_Open_Return'])
    total_weight = intraday_weight + close_to_open_weight
    df['Preliminary_Factor'] = (intraday_weight * df['Intraday_Return'] + close_to_open_weight * df['Close_to_Open_Return']) / total_weight
    
    # Integrate Volume Change
    df['Daily_Volume_Change'] = df['volume'] / df['volume'].shift(1)
    df['Adjusted_Preliminary_Factor'] = df['Preliminary_Factor'] * df['Daily_Volume_Change']
    
    # Calculate True Range (TR)
    df['Prev_Close'] = df['close'].shift(1)
    df['True_Range'] = df[['high' - 'low', 'high' - 'Prev_Close', 'Prev_Close' - 'low']].max(axis=1).abs()
    
    # Calculate Positive Directional Movement (+DM)
    df['+DM'] = (df['high'] > df['high'].shift(1)) & ((df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']))
    df['+DM'] = (df['high'] - df['high'].shift(1)).where(df['+DM'], 0)
    
    # Calculate Negative Directional Movement (-DM)
    df['-DM'] = (df['low'].shift(1) > df['low']) & ((df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)))
    df['-DM'] = (df['low'].shift(1) - df['low']).where(df['-DM'], 0)
    
    # Smooth +DM and -DM
    df['+DM_Smoothed'] = df['+DM'].ewm(span=14, min_periods=14).mean()
    df['-DM_Smoothed'] = df['-DM'].ewm(span=14, min_periods=14).mean()
    df['TR_Smoothed'] = df['True_Range'].ewm(span=14, min_periods=14).mean()
    
    # Calculate +DI and -DI
    df['+DI'] = 100 * (df['+DM_Smoothed'] / df['TR_Smoothed'])
    df['-DI'] = 100 * (df['-DM_Smoothed'] / df['TR_Smoothed'])
    
    # Calculate ADMI
    df['ADMI'] = (df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])
    
    # Incorporate Volume Weighted Returns
    df['Volume_Weighted_Return'] = (df['Intraday_Return'] * df['volume'] + df['Close_to_Open_Return'] * df['volume'])
    df['20Day_EMA_Volume_Weighted_Return'] = df['Volume_Weighted_Return'].ewm(span=20, min_periods=20).mean()
    
    # Synthesize Final Alpha Factor
    df['Difference'] = df['10Day_EMA_Price_Change'] - df['20Day_EMA_Volume_Weighted_Return']
    df['5Day_EMA_Difference'] = df['Difference'].ewm(span=5, min_periods=5).mean()
    df['Final_Alpha_Factor'] = df['5Day_EMA_Difference'] * df['ADMI']
    
    return df['Final_Alpha_Factor']
