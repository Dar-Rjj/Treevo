import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Close-to-Open Return
    df['Close_to_Open_Return'] = df['open'].shift(-1) - df['close']
    
    # Volume Weighting
    df['Volume_Weighted_Return'] = df['Close_to_Open_Return'] * df['volume']
    
    # Adaptive Window Calculation based on Volatility
    df['HLR'] = (df['high'] + df['low'] + df['close']) / 3
    df['Volatility'] = df['HLR'].rolling(window=30).std()
    
    # Define a function to determine the adaptive window size
    def get_adaptive_window(vol):
        if vol > df['Volatility'].quantile(0.75):
            return 5  # High volatility, decrease window
        else:
            return 30  # Low volatility, increase window
    
    # Apply the adaptive window size
    df['Adaptive_Window'] = df['Volatility'].apply(get_adaptive_window)
    
    # Calculate rolling statistics with adaptive window
    df['Rolling_Mean'] = df.groupby('date')['Volume_Weighted_Return'].transform(lambda x: x.rolling(window=int(x.name[1])).mean())
    df['Rolling_Std'] = df.groupby('date')['Volume_Weighted_Return'].transform(lambda x: x.rolling(window=int(x.name[1])).std())
    
    # Incorporate high-frequency data
    df['Intraday_Range'] = df['high'] - df['low']
    
    # Adjust the alpha factor
    df['Adjusted_Factor'] = df['Intraday_Range'] * (df['Volume_Weighted_Return'] - df['Rolling_Mean']) / df['Rolling_Std']
    
    # Combine factors and output as alpha factor
    df['Alpha_Factor'] = df['Adjusted_Factor'].fillna(0)
    
    return df['Alpha_Factor']

# Example usage:
# df = pd.read_csv('market_data.csv', parse_dates=['date'], index_col='date')
# alpha_factor = heuristics_v2(df)
# print(alpha_factor)
