import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Daily High-Low Range
    df['HL_Range'] = df['high'] - df['low']
    
    # Calculate 10-Day Sum of High-Low Ranges
    df['HL_10Day_Sum'] = df['HL_Range'].rolling(window=10).sum()
    
    # Calculate Price Change over 10 Days
    df['Price_Change_10D'] = df['close'] - df['close'].shift(10)
    
    # Calculate Price Momentum
    df['Close_Trend'] = df['close'].pct_change(periods=10)
    df['EMA_Close_Trend'] = df['Close_Trend'].ewm(span=21, adjust=False).mean()
    df['Momentum_Score'] = df['EMA_Close_Trend'].diff()
    df['WMA_Momentum_Score'] = df['Momentum_Score'].rolling(window=21, min_periods=1).apply(lambda x: np.average(x, weights=np.arange(1, len(x)+1)), raw=True)
    
    # Volume Confirmation
    df['Volume_Trend'] = df['volume'].pct_change(periods=10)
    df['EMA_Volume_Trend'] = df['Volume_Trend'].ewm(span=21, adjust=False).mean()
    df['Volume_Score'] = df['EMA_Volume_Trend'].diff()
    df['WMA_Volume_Score'] = df['Volume_Score'].rolling(window=21, min_periods=1).apply(lambda x: np.average(x, weights=np.arange(1, len(x)+1)), raw=True)
    
    # Incorporate Advanced Volatility
    df['True_Range'] = df[['high' - 'low', (df['high'] - df['close'].shift(1)).abs(), (df['low'] - df['close'].shift(1)).abs()]].max(axis=1)
    df['Open_Close_Volatility'] = (df['open'] - df['close']).abs()
    df['Combined_Volatility'] = df[['True_Range', 'Open_Close_Volatility']].max(axis=1)
    df['Smoothed_Combined_Volatility'] = df['Combined_Volatility'].ewm(span=21, adjust=False).mean()
    df['Volatility_Score'] = df['Smoothed_Combined_Volatility'].diff()
    df['WMA_Volatility_Score'] = df['Volatility_Score'].rolling(window=21, min_periods=1).apply(lambda x: np.average(x, weights=np.arange(1, len(x)+1)), raw=True)
    
    # Incorporate Price Pattern Analysis
    df['Inside_Bar'] = (df['high'] < df['high'].shift(1)) & (df['low'] > df['low'].shift(1))
    df['Outside_Bar'] = (df['high'] > df['high'].shift(1)) & (df['low'] < df['low'].shift(1))
    df['Inside_Bar_Score'] = df['Inside_Bar'].astype(int)
    df['Outside_Bar_Score'] = df['Outside_Bar'].astype(int)
    
    # Calculate Price Volatility
    df['Daily_Returns'] = df['close'].pct_change()
    df['Price_Volatility'] = df['Daily_Returns'].rolling(window=20).std()
    
    # Combine Scores
    df['Combined_Score'] = (df['WMA_Momentum_Score'] * df['WMA_Volume_Score']) + df['HL_10Day_Sum']
    df['Combined_Score_Adjusted'] = df['Combined_Score'] / df['Price_Volatility']
    df['Final_Score'] = df['Combined_Score_Adjusted'] / df['WMA_Volatility_Score'] + df['Inside_Bar_Score'] + df['Outside_Bar_Score']
    
    return df['Final_Score']
