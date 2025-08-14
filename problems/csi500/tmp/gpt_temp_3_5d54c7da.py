import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Price Momentum
    n_days = 21
    df['Close_Trend'] = df['close'].pct_change(periods=n_days)
    df['Close_EMA'] = df['Close_Trend'].ewm(span=21, adjust=False).mean()
    df['Momentum_Score'] = df['Close_EMA'].diff().rolling(window=21, min_periods=1).mean()

    # Volume Confirmation
    m_days = 21
    df['Volume_Trend'] = df['volume'].pct_change(periods=m_days)
    df['Volume_EMA'] = df['Volume_Trend'].ewm(span=21, adjust=False).mean()
    df['Volume_Score'] = df['Volume_EMA'].diff().rolling(window=21, min_periods=1).mean()

    # Incorporate Advanced Volatility
    df['High_Low_Volatility'] = df['high'] - df['low']
    df['Open_Close_Volatility'] = (df['open'] - df['close']).abs()
    df['Combined_Volatility'] = df[['High_Low_Volatility', 'Open_Close_Volatility']].max(axis=1)
    df['Volatility_EMA'] = df['Combined_Volatility'].ewm(span=21, adjust=False).mean()
    df['Volatility_Score'] = df['Volatility_EMA'].diff().rolling(window=21, min_periods=1).mean()

    # Calculate Daily High-Low Range and 10-Day Sum of High-Low Ranges
    df['High_Low_Range'] = df['high'] - df['low']
    df['10_Day_Sum_High_Low_Range'] = df['High_Low_Range'].rolling(window=10, min_periods=1).sum()

    # Incorporate Price Pattern Analysis
    df['Inside_Bar'] = ((df['high'] < df['high'].shift(1)) & (df['low'] > df['low'].shift(1))).astype(int)
    df['Outside_Bar'] = ((df['high'] > df['high'].shift(1)) & (df['low'] < df['low'].shift(1))).astype(int)

    # Combine All Scores
    df['Alpha_Factor'] = (df['Momentum_Score'] * df['Volume_Score'] + 
                          df['10_Day_Sum_High_Low_Range']) / df['Volatility_Score'] + 
                          df['Inside_Bar'] + df['Outside_Bar']

    return df['Alpha_Factor']
