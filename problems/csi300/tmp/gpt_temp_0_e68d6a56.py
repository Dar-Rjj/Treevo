import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Close-to-Open Return
    df['Close_to_Open_Return'] = df['open'].shift(-1) - df['close']
    
    # Volume Weighting
    df['Volume_Weighted_Return'] = df['Close_to_Open_Return'] * df['volume']
    
    # Determine Volatility
    df['True_Range'] = df[['high', 'low', 'close']].apply(lambda x: max(x) - min(x), axis=1)
    df['Volatility'] = df['True_Range'].rolling(window=20).std()
    
    # Define High and Low Volatility Thresholds
    avg_volatility = df['Volatility'].mean()
    high_volatility_threshold = 2 * avg_volatility
    low_volatility_threshold = 0.5 * avg_volatility
    
    # Adjust Window Size based on Volatility
    def adjust_window_size(volatility, avg_volatility, high_threshold, low_threshold, base_window=20):
        if volatility > high_threshold:
            return max(base_window // 2, 1)  # Decrease window size
        elif volatility < low_threshold:
            return min(base_window * 2, len(df))  # Increase window size
        else:
            return base_window
    
    df['Adaptive_Window'] = df['Volatility'].apply(lambda x: adjust_window_size(x, avg_volatility, high_volatility_threshold, low_volatility_threshold))
    
    # Rolling Statistics
    df['Rolling_Mean'] = df.apply(lambda row: df.loc[:row.name, 'Volume_Weighted_Return'].rolling(window=int(row['Adaptive_Window']), min_periods=1).mean().iloc[-1], axis=1)
    df['Rolling_Std'] = df.apply(lambda row: df.loc[:row.name, 'Volume_Weighted_Return'].rolling(window=int(row['Adaptive_Window']), min_periods=1).std().iloc[-1], axis=1)
    
    # Final Factor Calculation
    df['Factor'] = (df['Rolling_Mean'] - df['Rolling_Std']) / df['Rolling_Std']
    
    return df['Factor']
