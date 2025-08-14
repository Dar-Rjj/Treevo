import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate 14-Period Exponential Moving Averages
    df['High_EMA_14'] = df['high'].ewm(span=14, adjust=False).mean()
    df['Low_EMA_14'] = df['low'].ewm(span=14, adjust=False).mean()
    df['Close_EMA_14'] = df['close'].ewm(span=14, adjust=False).mean()
    df['Open_EMA_14'] = df['open'].ewm(span=14, adjust=False).mean()
    
    # Compute 14-Period Price Envelopes
    df['Max_Price'] = df[['high', 'close']].max(axis=1)
    df['Min_Price'] = df[['low', 'close']].min(axis=1)
    df['Envelope_Distance'] = df['Max_Price'] - df['Min_Price']
    df['Volume_Weighted_Envelope'] = (df['Envelope_Distance'] * df['volume']).ewm(span=14, adjust=False).mean()
    
    # Construct Momentum Oscillator
    df['Smoothed_Positive_Momentum'] = ((df['High_EMA_14'] - df['Close_EMA_14']) * df['volume']).apply(lambda x: x if x > 0 else 0)
    df['Smoothed_Negative_Momentum'] = ((df['Low_EMA_14'] - df['Close_EMA_14']) * df['volume']).apply(lambda x: x if x < 0 else 0)
    df['Momentum_Indicator'] = df['Smoothed_Positive_Momentum'] - df['Smoothed_Negative_Momentum']
    
    # Calculate Volume-Weighted Average True Range
    df['True_Range'] = df.apply(lambda row: max(row['high'] - row['low'], abs(row['high'] - df.shift(1).loc[row.name, 'close']), abs(row['low'] - df.shift(1).loc[row.name, 'close'])), axis=1)
    df['ATR'] = df['True_Range'].ewm(span=14, adjust=False).mean()
    df['Volume_Weighted_ATR'] = (df['ATR'] * df['volume']).ewm(span=14, adjust=False).mean()
    
    # Final Alpha Factor
    df['Alpha_Factor'] = df['Momentum_Indicator'] / df['Volume_Weighted_ATR']
    df['Alpha_Factor'] = df['Alpha_Factor'].where(df['Alpha_Factor'].abs() > 0.5, 0)  # Apply a threshold to filter out weak signals
    
    return df['Alpha_Factor']
