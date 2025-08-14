import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Close-to-Open Return
    df['Close_to_Open_Return'] = df['open'].shift(-1) - df['close']
    
    # Volume Weighting
    df['Volume_Weighted_Return'] = df['Volume_Weighted_Return'] * df['volume']
    
    # Determine Volatility (using High, Low, and Close prices)
    df['Volatility'] = df[['high', 'low', 'close']].rolling(window=20).std().mean(axis=1)
    
    # Adjust Window Size based on Volatility
    def adjust_window(volatility):
        if volatility > df['Volatility'].median():
            return 10
        else:
            return 30
    
    df['Rolling_Window_Size'] = df['Volatility'].apply(adjust_window)
    
    # Calculate Rolling Statistics with Adaptive Window
    rolling_mean = df.groupby('Rolling_Window_Size')['Volume_Weighted_Return'].transform(lambda x: x.rolling(window=x.name).mean())
    rolling_std = df.groupby('Rolling_Window_Size')['Volume_Weighted_Return'].transform(lambda x: x.rolling(window=x.name).std())
    
    # Combine Rolling Mean and Standard Deviation for final factor
    df['Alpha_Factor'] = (rolling_mean - rolling_std) / rolling_std
    
    # Return the Alpha Factor
    return df['Alpha_Factor']
