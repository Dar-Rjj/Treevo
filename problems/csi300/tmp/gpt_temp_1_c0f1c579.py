import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Daily Price Change
    df['Daily_Price_Change'] = df['close'].diff()
    
    # Calculate 5-Day Price Momentum
    df['5_Day_Price_Momentum'] = df['Daily_Price_Change'].rolling(window=5).sum()
    df['5_Day_Price_Momentum'] = df['5_Day_Price_Momentum'].apply(lambda x: x if x != 0 else 1)
    
    # Calculate Volume Confirmation
    df['Volume_Confirmation'] = (df['volume'] > df['volume'].shift(1)).astype(int)
    df['Volume_Confirmation'] *= df['5_Day_Price_Momentum']
    
    # Calculate Amplitude Confirmation
    df['Daily_Amplitude'] = df['high'] - df['low']
    df['Amplitude_Confirmation'] = (df['Daily_Amplitude'] > df['Daily_Amplitude'].shift(1)).astype(int)
    df['Amplitude_Confirmation'] *= df['5_Day_Price_Momentum']
    
    # Adjust for Volatility
    df['True_Range'] = df[['high' - 'low', 
                            (df['high'] - df['close'].shift(1)).abs(), 
                            (df['low'] - df['close'].shift(1)).abs()]].max(axis=1)
    df['5_Day_ATR'] = df['True_Range'].rolling(window=5).mean()
    df['Adjusted_Confirmation'] = (df['Volume_Confirmation'] + df['Amplitude_Confirmation']) / df['5_Day_ATR']
    
    # Introduce Trend Strength Confirmation
    df['20_Day_SMA'] = df['close'].rolling(window=20).mean()
    df['Trend_Strength_Confirmation'] = (df['close'] > df['20_Day_SMA']).astype(int)
    df['Alpha_Factor'] = df['Trend_Strength_Confirmation'] * df['Adjusted_Confirmation']
    
    return df['Alpha_Factor']
