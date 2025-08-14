import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Parameters
    n = 20
    m = 10
    
    # Compute Raw Momentum
    df['Raw_Momentum'] = df['close'] - df['close'].shift(n)
    
    # Adjust for Volume
    df['Average_Volume'] = df['volume'].rolling(window=n).mean()
    df['Volume_Ratio'] = df['volume'] / df['Average_Volume']
    df['Adjusted_Raw_Momentum'] = df['Raw_Momentum'] * df['Volume_Ratio']
    
    # Incorporate True Range Momentum
    df['True_Range'] = df.apply(lambda x: max(abs(x['high'] - x['low']), 
                                              abs(x['high'] - x['close'].shift(1)), 
                                              abs(x['low'] - x['close'].shift(1))), axis=1)
    df['ATR'] = df['True_Range'].ewm(span=n, adjust=False).mean()
    df['True_Range_Momentum'] = df['Raw_Momentum'] / df['ATR']
    
    # Combine Raw and True Range Momentum
    df['Combined_Momentum'] = df['Adjusted_Raw_Momentum'] + df['True_Range_Momentum']
    
    # Adjust by Volume Confirmation
    df['5D_Volume_EMA'] = df['volume'].ewm(span=5, adjust=False).mean()
    df['20D_Volume_EMA'] = df['volume'].ewm(span=20, adjust=False).mean()
    df['Volume_Confirmation'] = np.where(df['5D_Volume_EMA'] > df['20D_Volume_EMA'], 1.2, 0.8)
    df['Combined_Momentum_Adjusted'] = df['Combined_Momentum'] * df['Volume_Confirmation']
    
    # Incorporate Enhanced Gaps and Oscillations
    df['Open_to_Close_Gap'] = df['open'] - df['close']
    df['High_Low_Gap'] = df['high'] - df['low']
    df['Historical_Average_Volume'] = df['volume'].rolling(window=m).mean()
    df['Volume_Difference'] = df['volume'] - df['Historical_Average_Volume']
    df['Volume_Oscillation'] = df['Volume_Difference'] / df['Historical_Average_Volume']
    
    # Final Factor
    df['Final_Factor'] = (df['Combined_Momentum_Adjusted'] 
                          + df['Open_to_Close_Gap'] 
                          + df['High_Low_Gap'] 
                          + df['Volume_Oscillation'])
    
    return df['Final_Factor']
