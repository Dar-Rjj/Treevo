import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Close-to-Open Return
    df['Close_to_Open_Return'] = df['open'].shift(-1) - df['close']
    
    # Volume Weighting
    df['Volume_Weighted_Return'] = df['Close_to_Open_Return'] * df['volume']
    
    # Determine Volatility using High, Low, and Close prices
    df['Volatility'] = df[['high', 'low', 'close']].rolling(window=30).std().mean(axis=1)
    
    # Adjust Window Size based on Volatility
    def adaptive_window(v):
        if v > df['Volatility'].quantile(0.75):
            return 5  # Decrease window size for high volatility
        else:
            return 60  # Increase window size for low volatility
    
    df['Adaptive_Window'] = df['Volatility'].apply(adaptive_window)
    
    # Calculate Rolling Statistics with Adaptive Window
    rolling_mean = df.groupby('Adaptive_Window')['Volume_Weighted_Return'].transform(lambda x: x.rolling(window=x.name, min_periods=1).mean())
    rolling_std = df.groupby('Adaptive_Window')['Volume_Weighted_Return'].transform(lambda x: x.rolling(window=x.name, min_periods=1).std())
    
    # Final Alpha Factor: Standardized Volume Weighted Close-to-Open Return
    df['Alpha_Factor'] = (rolling_mean - rolling_mean.mean()) / rolling_std
    
    return df['Alpha_Fctor'].dropna()
