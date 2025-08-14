import pandas as pd
import pandas as pd

def heuristics_v2(df, n=14, ema_short_window=9):
    # Calculate True Range and Daily Gaps
    df['True Range'] = df.apply(lambda row: max(row['High'] - row['Low'], 
                                                abs(row['High'] - df.shift(1).loc[row.name, 'Close']), 
                                                abs(row['Low'] - df.shift(1).loc[row.name, 'Close'])), axis=1)
    df['Daily Gap'] = df['Open'] - df.shift(1)['Close']
    
    # Compute Average True Range (ATR) and Volume Weighted Average of Gaps
    df['ATR'] = df['True Range'].rolling(window=n).mean()
    df['Volume Weighted Gap'] = (df['Daily Gap'] * df['Volume']).sum() / df['Volume'].sum()
    
    # Calculate Price Momentum and Exponential Moving Average (EMA)
    df['Price Momentum'] = df['Close'] - df['Close'].shift(n)
    df['EMA'] = df['Close'].ewm(span=ema_short_window, adjust=False).mean()
    df['EMA Change'] = df['EMA'] - df['EMA'].shift(1)
    
    # Adjust Momentum and EMA Change by ATR and Volume
    df['Adjusted Momentum'] = df['Price Momentum'] / df['ATR']
    df['Final Alpha Factor'] = (df['Adjusted Momentum'] * df['Volume Weighted Gap'] * df['EMA Change'])
    
    # Introduce Volume Adjustment
    df['Volume MA'] = df['Volume'].rolling(window=n).mean()
    df['Final Alpha Factor'] = df['Final Alpha Factor'] * df['Volume MA']
    
    return df['Final Alpha Factor']
