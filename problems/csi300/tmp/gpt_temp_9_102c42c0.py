import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Close-to-Open Return
    df['Close_to_Open_Return'] = df['open'].shift(-1) - df['close']
    
    # Volume Weighting
    df['Volume_Weighted_Return'] = df['Close_to_Open_Return'] * df['volume']
    
    # Dynamic Volatility Calculation
    df['High_Low_Close_Avg'] = (df['high'] + df['low'] + df['close']) / 3
    fixed_vol_window = 20  # Fixed window for initial volatility calculation
    df['Volatility'] = df['High_Low_Close_Avg'].rolling(window=fixed_vol_window).std()
    
    # Adaptive Window Sizing
    median_volatility = df['Volatility'].median()
    df['Adaptive_Window'] = np.where(df['Volatility'] > median_volatility, 10, 30)
    
    # Rolling Statistics with Adaptive Window
    df['Rolling_Mean'] = df['Volume_Weighted_Return'].rolling(window=df['Adaptive_Window']).mean()
    df['Rolling_Std'] = df['Volume_Weighted_Return'].rolling(window=df['Adaptive_Window']).std()
    
    # Final Alpha Factor
    df['Normalized_Factor'] = (df['Volume_Weighted_Return'] - df['Rolling_Mean']) / df['Rolling_Std']
    
    return df['Normalized_Factor'].dropna()

# Example usage:
# df = pd.read_csv('your_data.csv', parse_dates=True, index_col='date')
# alpha_factor = heuristics_v2(df)
# print(alpha_factor)
