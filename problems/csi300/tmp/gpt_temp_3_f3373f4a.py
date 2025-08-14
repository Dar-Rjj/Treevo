import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Close-to-Open Return
    df['Close_to_Open_Return'] = df['open'].shift(-1) - df['close']
    
    # Volume Weighting
    df['Volume_Weighted_Return'] = df['Close_to_Open_Return'] * df['volume']
    
    # Determine Volatility using High, Low, and Close prices up to day t
    df['True_Range'] = df[['high', 'low', 'close']].apply(lambda x: max(x) - min(x), axis=1)
    volatility_window = 30  # Fixed window for calculating volatility
    df['Volatility'] = df['True_Range'].rolling(window=volatility_window).std()
    
    # Adjust Window Size based on Volatility
    base_window = 60  # Base window size
    volatility_threshold_high = df['Volatility'].quantile(0.75)
    volatility_threshold_low = df['Volatility'].quantile(0.25)
    
    def adjust_window(volatility):
        if volatility > volatility_threshold_high:
            return int(base_window * 0.8)  # Decrease window size
        elif volatility < volatility_threshold_low:
            return int(base_window * 1.2)  # Increase window size
        else:
            return base_window
    
    df['Window_Size'] = df['Volatility'].apply(adjust_window)
    
    # Rolling Statistics with Adaptive Window
    rolling_mean = df['Volume_Weighted_Return'].rolling(window=df['Window_Size']).mean()
    rolling_std = df['Volume_Weighted_Return'].rolling(window=df['Window_Size']).std()
    
    # Final Alpha Factor
    df['Alpha_Factor'] = (rolling_mean - df['Volume_Weighted_Return']) / rolling_std
    
    # Return the alpha factor as a pandas Series
    return df['Alpha_Factor'].dropna()
