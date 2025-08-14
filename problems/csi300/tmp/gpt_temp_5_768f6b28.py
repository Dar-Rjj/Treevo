import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Raw Momentum
    df['Raw_Momentum'] = df['close'] / df['close'].shift(20) - 1
    
    # Calculate High-Low Range
    df['High_Low_Range'] = df['high'] - df['low']
    
    # Calculate True Range
    df['True_Range'] = df[['high', 'low']].apply(
        lambda x: max(x['high'] - x['low'], 
                      abs(x['high'] - df['close'].shift(1)), 
                      abs(x['low'] - df['close'].shift(1))), axis=1)
    
    # Calculate Price Volatility
    df['Log_Returns'] = np.log(df['close'] / df['close'].shift(1))
    df['Price_Volatility'] = df['Log_Returns'].rolling(window=20).apply(lambda x: (x**2).sum(), raw=True)
    
    # Calculate Volume Trend
    df['Volume_Trend'] = df['volume'].rolling(window=5).mean() / df['volume'].rolling(window=20).mean()
    
    # Combine Components
    df['Factor'] = (df['Raw_Momentum'] * df['True_Range'] * 
                    np.sqrt(df['Price_Volatility']) * df['Volume_Trend'])
    
    # Apply 5-day Exponential Moving Average
    df['Factor'] = df['Factor'].ewm(span=5, adjust=False).mean()
    
    return df['Factor']
