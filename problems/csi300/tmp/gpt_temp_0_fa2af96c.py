import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Close-to-Open Return
    df['Close_to_Open_Return'] = df['open'].shift(-1) - df['close']
    
    # Volume Weighting
    df['Volume_Weighted_Return'] = df['Close_to_Open_Return'] * df['volume']
    
    # Volatility Calculation using High, Low, and Close prices
    df['HL_C_Volatility'] = df[['high', 'low', 'close']].std(axis=1)
    
    # Adaptive Window Size based on volatility
    volatility_median = df['HL_C_Volatility'].median()
    window_size = df['HL_C_Volatility'].apply(lambda x: 5 if x > volatility_median else 20)
    
    # Rolling Statistics with adaptive window size
    def rolling_stat(ser, win_ser, func):
        return ser.groupby(
            lambda x: (x - ser.index[0]).days // win_ser.iloc[x]
        ).transform(func).values
    
    df['Rolling_Mean'] = rolling_stat(df['Volume_Weighted_Return'], window_size, 'mean')
    df['Rolling_Std'] = rolling_stat(df['Volume_Weighted_Return'], window_size, 'std')
    
    # Final Alpha Factor
    df['Normalized_Volume_Weighted_Return'] = (
        df['Volume_Weighted_Return'] - df['Rolling_Mean']
    ) / df['Rolling_Std']
    
    return df['Normalized_Volume_Weighted_Return'].dropna()

# Example usage:
# Assume `df` is a pandas DataFrame with the following columns: ['date', 'open', 'high', 'low', 'close', 'amount', 'volume']
# df.set_index('date', inplace=True)
# alpha_factor = heuristics_v2(df)
