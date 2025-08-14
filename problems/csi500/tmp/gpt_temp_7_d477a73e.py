import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Exponential Moving Averages (EMAs)
    df['EMA_5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()

    # Determine Dynamic Momentum
    df['Momentum'] = df['EMA_5'] - df['EMA_20']
    df['Sign_Momentum'] = np.sign(df['Momentum'])

    # Calculate Price Range
    df['True_Range'] = 0.0
    df['True_Range'] = df.apply(lambda x: max(x['high'] - x['low'], 
                                              abs(x['high'] - df.shift(1).loc[x.name, 'close']), 
                                              abs(x['low'] - df.shift(1).loc[x.name, 'close'])), axis=1)
    
    df['Sum_True_Range_14'] = df['True_Range'].rolling(window=14).sum()

    # Compute Average True Range (ATR)
    df['ATR'] = df['Sum_True_Range_14'] / 14

    # Combine Dynamic Momentum and Volatility
    df['Momentum_ATR'] = df['Sign_Momentum'] * df['ATR']
    df['Weighted_Momentum_Vol'] = df['Momentum_ATR'] * df['volume']

    # Integrate Market Trend Strength
    df['Long_Term_Momentum'] = df['EMA_20'] - df['EMA_50']
    df['Sign_Long_Term_Momentum'] = np.sign(df['Long_Term_Momentum'])

    # Combine Short-Term and Long-Term Momentum
    df['Combined_Momentum'] = df['Sign_Momentum'] + df['Sign_Long_Term_Momentum']
    df['Adjusted_Momentum_Vol'] = df['Combined_Momentum'] * df['ATR'] * df['volume']

    # Adjust for Recent Price Change
    df['Price_Change'] = df['close'] - df['close'].shift(1)
    df['Final_Alpha_Factor'] = df['Adjusted_Momentum_Vol'] * df['Price_Change']

    return df['Final_Alpha_Factor']
