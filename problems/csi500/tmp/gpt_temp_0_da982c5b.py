import pandas as pd
import numpy as np
import pandas as pd

def heuristics_v2(df, n=20):
    # Calculate Daily Price Movement
    df['Price_Movement'] = df['close'] - df['open']
    
    # Identify Price Breakaways
    df['High_Breakaway'] = (df['high'] > df['high'].shift(1)) & ((df['high'] - df['high'].shift(1)) / df['high'].shift(1) > 0.05)
    df['Low_Breakaway'] = (df['low'] < df['low'].shift(1)) & ((df['low'].shift(1) - df['low']) / df['low'].shift(1) > 0.05)
    
    # Volume-Weighted Score
    conditions = [
        (df['High_Breakaway'] & ~df['Low_Breakaway']),
        (df['Low_Breakaway'] & ~df['High_Breakaway'])
    ]
    choices = [
        df['Price_Movement'] * df['volume'],
        -(df['Price_Movement'] * df['volume'])
    ]
    df['Volume_Weighted_Score'] = pd.np.select(conditions, choices, default=0)
    
    # Calculate High-to-Low Price Percentage Change
    df['High_to_Low_Percentage_Change'] = (df['high'] - df['low']) / df['low']
    
    # Combine Volume-Weighted Score with High-to-Low Percentage Change
    df['Combined_Score'] = 0
    df.loc[df['Volume_Weighted_Score'] != 0, 'Combined_Score'] = df['Volume_Weighted_Score'] + (df['High_to_Low_Percentage_Change'] * df['volume'])
    
    # Final Alpha Factor
    df['Final_Alpha_Factor'] = df['Combined_Score'].rolling(window=n).sum()
    
    return df['Final_Alpha_Factor']
