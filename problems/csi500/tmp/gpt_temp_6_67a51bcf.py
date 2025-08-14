import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate EMAs
    df['5_day_EMA'] = df['close'].ewm(span=5, adjust=False).mean()
    df['20_day_EMA'] = df['close'].ewm(span=20, adjust=False).mean()
    
    # Determine Price Momentum
    df['Momentum'] = df['5_day_EMA'] - df['20_day_EMA']
    df['Momentum_Sign'] = np.sign(df['Momentum'])
    
    # Calculate True Range
    df['True_Range'] = df.apply(lambda x: max(x['high'] - x['low'], 
                                               abs(x['high'] - df.shift(1).loc[x.name, 'close']), 
                                               abs(x['low'] - df.shift(1).loc[x.name, 'close'])), axis=1)
    
    # Compute Average True Range (ATR)
    df['ATR'] = df['True_Range'].rolling(window=14).mean()
    
    # Exponential Smoothing of Momentum
    alpha = 0.2  # Smoothing factor
    df['Smoothed_Momentum'] = df['Momentum'].ewm(alpha=alpha, adjust=False).mean()
    
    # Combine Momentum and Volatility
    df['Combined_Factor'] = df['Smoothed_Momentum'] * df['ATR']
    
    # Weight by Volume
    df['Alpha_Factor'] = df['Combined_Factor'] * df['volume']
    
    return df['Alpha_Factor']
