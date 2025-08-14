import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Daily Price Change
    df['Daily_Price_Change'] = df['close'].diff()
    
    # Compute Volume-Adjusted Return
    df['Volume_Adjusted_Return'] = df['Daily_Price_Change'] / (df['volume'] + 1)
    
    # Initialize Gain and Loss Arrays
    df['Gain'] = 0.0
    df['Loss'] = 0.0
    
    # Calculate Rolling Average Volume
    df['Rolling_Avg_Volume'] = df['volume'].rolling(window=10).mean()
    
    # Calculate Volume-Adjusted Momentum
    for i in range(1, len(df)):
        if df.loc[df.index[i], 'Volume_Adjusted_Return'] > 0:
            df.loc[df.index[i], 'Gain'] = df.loc[df.index[i], 'Volume_Adjusted_Return']
            df.loc[df.index[i], 'Loss'] = 0.0
        else:
            df.loc[df.index[i], 'Loss'] = abs(df.loc[df.index[i], 'Volume_Adjusted_Return'])
            df.loc[df.index[i], 'Gain'] = 0.0
    
    # Calculate Relative Strength
    df['Avg_Gain'] = df['Gain'].rolling(window=14).mean()
    df['Avg_Loss'] = df['Loss'].rolling(window=14).mean()
    df['Relative_Strength'] = df['Avg_Gain'] / (df['Avg_Loss'] + 1e-6)  # Avoid division by zero
    
    # Calculate RSI from Relative Strength
    df['RSI'] = 100 - (100 / (1 + df['Relative_Strength']))
    
    # Calculate Volume-Adjusted Momentum Indicator
    df['Volume_Adjusted_Momentum_Indicator'] = df['RSI'] * df['Rolling_Avg_Volume']
    
    # Output the final alpha factor
    return df['Volume_Adjusted_Momentum_Indicator']

# Example usage:
# df = pd.read_csv('your_data.csv', index_col='date', parse_dates=True)
# factor_values = heuristics_v2(df)
# print(factor_values)
