import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Simple Moving Average (SMA) crossover
    short_window = 5
    long_window = 20
    df['SMA_5'] = df['close'].rolling(window=short_window).mean()
    df['SMA_20'] = df['close'].rolling(window=long_window).mean()
    df['SMA_Crossover'] = df['SMA_5'] - df['SMA_20']
    
    # Price Rate of Change (ROC)
    roc_period = 14
    df['ROC'] = df['close'].pct_change(periods=roc_period)
    
    # On-Balance Volume (OBV)
    df['OBV'] = 0
    close_diff = df['close'].diff()
    for i in range(1, len(df)):
        if close_diff[i] > 0:
            df.loc[df.index[i], 'OBV'] = df['OBV'][i-1] + df['volume'][i]
        elif close_diff[i] < 0:
            df.loc[df.index[i], 'OBV'] = df['OBV'][i-1] - df['volume'][i]
        else:
            df.loc[df.index[i], 'OBV'] = df['OBV'][i-1]
    
    # Volume-Price Trend (VPT)
    df['VPT'] = ((df['close'] - df['close'].shift(1)) / df['close'].shift(1)) * df['volume']
    df['VPT'] = df['VPT'].cumsum()
    
    # Average True Range (ATR)
    atr_period = 14
    df['TR'] = df[['high', 'close']].max(axis=1) - df[['low', 'close']].min(axis=1)
    df['ATR'] = df['TR'].rolling(window=atr_period).mean()
    
    # Engulfing Pattern
    df['Bullish_Engulfing'] = ((df['open'] < df['close'].shift(1)) & (df['close'] > df['open'].shift(1)) & (df['open'] < df['close']))
    df['Bearish_Engulfing'] = ((df['open'] > df['close'].shift(1)) & (df['close'] < df['open'].shift(1)) & (df['open'] > df['close']))

    # Combine multiple factors to create a composite alpha factor
    df['Alpha_Factor'] = df['SMA_Crossover'] + df['ROC'] + df['OBV'] + df['VPT'] + df['ATR']
    df['Alpha_Factor'] = df['Alpha_Factor'].where(~df['Bullish_Engulfing'], other=df['Alpha_Factor'] * 1.1)
    df['Alpha_Factor'] = df['Alpha_Factor'].where(~df['Bearish_Engulfing'], other=df['Alpha_Factor'] * 0.9)
    
    return df['Alpha_Factor']
