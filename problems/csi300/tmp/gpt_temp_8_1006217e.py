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
    df['HL_C_mean'] = (df['high'] + df['low'] + df['close']) / 3
    df['Volatility'] = df['HL_C_mean'].rolling(window=20).std()
    
    # Adaptive Window Calculation
    def adjust_window(volatility, default_window=20):
        if volatility > np.percentile(df['Volatility'], 75):
            return max(5, default_window - 5)
        else:
            return min(30, default_window + 5)

    df['Adaptive_Window'] = df['Volatility'].apply(adjust_window)
    
    # Rolling Statistics with Adaptive Window
    df['Rolling_Mean'] = df.groupby('Volume_Weighted_Return')['Volume_Weighted_Return'].apply(
        lambda x: x.rolling(window=x.name['Adaptive_Window']).mean())
    df['Rolling_Std'] = df.groupby('Volume_Weighted_Return')['Volume_Weighted_Return'].apply(
        lambda x: x.rolling(window=x.name['Adaptive_Window']).std())
    
    # Incorporate Intraday Data
    df['Intraday_Range'] = df['high'] - df['low']
    df['Adjusted_Factor'] = (df['Volume_Weighted_Return'] - df['Rolling_Mean']) / df['Rolling_Std'] * df['Intraday_Range']
    
    # Combine Factors
    df['Alpha_Factor'] = df['Adjusted_Factor']

    return df['Alpha_Factor'].dropna()

# Example usage:
# df = pd.read_csv('your_data.csv', parse_dates=['date'], index_col='date')
# alpha_factor = heuristics_v2(df)
# print(alpha_factor)
