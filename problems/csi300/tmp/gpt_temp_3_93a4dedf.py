import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Daily Intraday Returns
    df['High-Low'] = df['high'] - df['low']
    df['Normalized_High_Low'] = df['High-Low'] / df['open']

    # Incorporate Volume Adjusted Momentum
    df['Close-to-Open_Return'] = (df['close'] - df['open']) / df['open']
    df['Volume_Adjusted_Momentum'] = df['Close-to-Open_Return'] * df['volume']

    # Integrate Volatility
    df['True_Range'] = df[['high' - 'low', 
                           abs('high' - df['close'].shift(1)), 
                           abs('low' - df['close'].shift(1))]].max(axis=1)
    df['ATR_14'] = df['True_Range'].rolling(window=14).mean()
    df['Intraday_Returns_ATR_Normalized'] = df['Normalized_High_Low'] / df['ATR_14']

    # Incorporate Exponential Moving Averages
    df['EMA_5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['EMA_Trend_Signal'] = df['EMA_5'] - df['EMA_20']

    # Combine Factors into Final Alpha
    df['Alpha'] = (df['Intraday_Returns_ATR_Normalized'] + 
                   df['Volume_Adjusted_Momentum'] + 
                   np.sign(df['EMA_Trend_Signal']))

    return df['Alpha']
