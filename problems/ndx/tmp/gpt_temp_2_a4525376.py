import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate the 10-day lowest low
    df['Lowest_Low_10'] = df['low'].rolling(window=10).min()
    # Calculate the close-to-lowest-low ratio
    df['Close_to_Lowest_Low_Ratio'] = df['close'] / df['Lowest_Low_10']
    # Calculate daily return
    df['Daily_Return'] = df['close'].pct_change()
    # Adjust daily return with volume
    df['Volume_Adjusted_Return'] = df['Daily_Return'] * (df['volume'] / df['volume'].ewm(span=5, adjust=False).mean())
    # Exponential moving average of the adjusted return
    df['EMA_Volume_Adjusted_Return'] = df['Volume_Adjusted_Return'].ewm(span=10, adjust=False).mean()
    # Construct the heuristics matrix
    heuristics_matrix = pd.Series(df['Close_to_Lest_Low_Ratio'] + df['EMA_Volume_Adjusted_Return'], name='heuristic_factor').dropna()
    
    return heuristics_matrix
