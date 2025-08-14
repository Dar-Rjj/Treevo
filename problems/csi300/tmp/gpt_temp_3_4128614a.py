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
    df['HL_C_volatility'] = df[['high', 'low', 'close']].std(axis=1)
    initial_window = 20  # Initial fixed window for volatility calculation
    df['volatility_std'] = df['HL_C_volatility'].rolling(window=initial_window).std()
    
    # Adjust Window Size based on Volatility
    def adjust_window(std):
        if std > df['HL_C_volatility'].mean():
            return max(5, int(initial_window * 0.8))  # Decrease window size
        else:
            return min(30, int(initial_window * 1.2))  # Increase window size
    
    df['adjusted_window'] = df['volatility_std'].apply(adjust_window)
    
    # Rolling Mean of Volume
    df['volume_mean'] = df['volume'].rolling(window=df['adjusted_window']).mean()
    
    # Enhanced Volatility Adjustment
    def adjust_return(volume, volume_mean, return_value):
        if volume > volume_mean:
            return return_value * 1.2  # Increase the absolute value of return
        else:
            return return_value * 0.8  # Decrease the absolute value of return
    
    df['Adjusted_Volume_Weighted_Return'] = [adjust_return(v, vm, r) for v, vm, r in zip(df['volume'], df['volume_mean'], df['Volume_Weighted_Return'])]
    
    # Rolling Statistics
    df['Rolling_Mean'] = df.groupby('adjusted_window')['Adjusted_Volume_Weighted_Return'].transform(lambda x: x.rolling(window=x.name).mean())
    df['Rolling_STD'] = df.groupby('adjusted_window')['Adjusted_Volume_Weighted_Return'].transform(lambda x: x.rolling(window=x.name).std())
    
    # Final Alpha Factor
    df['Alpha_Factor'] = (df['Adjusted_Volume_Weighted_Return'] - df['Rolling_Mean']) / df['Rolling_STD']
    
    return df['Alpha_Factor']
