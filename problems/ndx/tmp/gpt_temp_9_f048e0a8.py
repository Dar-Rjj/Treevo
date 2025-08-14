import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate 50-day Simple Moving Average (SMA)
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    
    # Calculate 20-day Exponential Moving Average (EMA)
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
    
    # Generate trend reversal signals
    df['Bullish_Signal'] = (df['EMA_20'] > df['SMA_50']) & (df['EMA_20'].shift(1) <= df['SMA_50'].shift(1))
    df['Bearish_Signal'] = (df['EMA_20'] < df['SMA_50']) & (df['EMA_20'].shift(1) >= df['SMA_50'].shift(1))
    
    # Calculate True Range (TR)
    df['TR'] = df[['high', 'low', 'close']].apply(lambda x: max(x['high'] - x['low'], abs(x['high'] - x['close'].shift(1)), abs(x['low'] - x['close'].shift(1))), axis=1)
    
    # Calculate 14-day Average True Range (ATR)
    df['ATR_14'] = df['TR'].rolling(window=14).mean()
    
    # Calculate On-Balance Volume (OBV)
    df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    
    # Calculate Volume Weighted Average Price (VWAP)
    df['VWAP'] = (df[['high', 'low', 'close']].mean(axis=1) * df['volume']).cumsum() / df['volume'].cumsum()
    
    # Identify days with volume greater than 2x the 20-day average volume
    df['Volume_Spike'] = df['volume'] > df['volume'].rolling(window=20).mean() * 2
    
    # Composite Trend Indicator
    df['Trend_Score'] = (df['Bullish_Signal'] * 1) - (df['Bearish_Signal'] * 1)
    
    # Risk-Adjusted Return Indicator (Sharpe Ratio)
    df['Daily_Return'] = df['close'].pct_change()
    df['Sharpe_Ratio'] = (df['Daily_Return'].rolling(window=20).mean()) / df['ATR_14']
    
    # Momentum and Reversal Factor
    df['EMA_20_Slope'] = df['EMA_20'].diff()
    df['Momentum_Factor'] = df['EMA_20_Slope'] + df['OBV'].diff()
    
    # Final Alpha Factor
    df['Alpha_Factor'] = df['Trend_Score'] + df['Sharpe_Ratio'] + df['Momentum_Factor']
    
    return df['Alpha_Factor']
