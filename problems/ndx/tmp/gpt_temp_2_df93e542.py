import pandas as pd
import numpy as np
import pandas as pd

def heuristics_v2(df):
    # Calculate Exponential Momentum
    df['Exponential_Momentum'] = (df['close'] - df['close'].shift(30)) / df['close'].shift(30)
    
    # Calculate 30-Day Average True Range for Stability
    df['True_Range'] = df.apply(lambda x: max(x['high'] - x['low'], x['high'] - df['close'].shift(1).loc[x.name], df['close'].shift(1).loc[x.name] - x['low']), axis=1)
    df['30D_Avg_True_Range'] = df['True_Range'].rolling(window=30).mean()
    
    # Calculate Short-Term and Long-Term Exponential Moving Averages
    df['50D_EMA'] = df['close'].ewm(span=50, adjust=False).mean()
    df['200D_EMA'] = df['close'].ewm(span=200, adjust=False).mean()
    
    # Adjust Momentum Factor with Inverse of Volatility
    df['Momentum_Adjusted_Volatility'] = df['Exponential_Momentum'] * (1 / df['30D_Avg_True_Range'])
    
    # Incorporate Volume into the Momentum Adjustment
    df['Volume_Percentage_30D_Avg'] = df['volume'] / df['volume'].rolling(window=30).mean()
    df['Momentum_Adjusted_Volume'] = df['Momentum_Adjusted_Volatility'] * df['Volume_Percentage_30D_Avg']
    
    # Integrate Price Position Relative to Exponential Moving Averages
    conditions = [
        (df['close'] > df['50D_EMA']) & (df['close'] > df['200D_EMA']),
        (df['close'] < df['50D_EMA']) & (df['close'] < df['200D_EMA']),
    ]
    choices = [1, -1]
    df['Position_Relative_to_EMAs'] = pd.np.select(conditions, choices, default=0)
    
    # Final Alpha Factor Composition
    df['Alpha_Factor'] = df['Momentum_Adjusted_Volume'] + df['Position_Relative_to_EMAs']
    
    return df['Alpha_Factor']
