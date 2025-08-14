import pandas as pd
import numpy as np
import pandas as pd

def heuristics_v2(df):
    # Calculate rate of change over 5 and 20 days
    df['ROC_5'] = (df['close'] / df['close'].shift(5)) - 1
    df['ROC_20'] = (df['close'] / df['close'].shift(20)) - 1
    
    # Calculate Williams %R over 14 days
    high_14 = df['high'].rolling(window=14).max()
    low_14 = df['low'].rolling(window=14).min()
    df['Williams_R_14'] = (high_14 - df['close']) / (high_14 - low_14) * -100
    
    # Calculate Volatility over 30 days
    df['Volatility_30'] = df['close'].rolling(window=30).std()
    
    # Calculate Average Volume over 10 and 20 days, and Volume Relative Strength
    df['Avg_Volume_10'] = df['volume'].rolling(window=10).mean()
    df['Avg_Volume_20'] = df['volume'].rolling(window=20).mean()
    df['VRS'] = df['Avg_Volume_10'] / df['Avg_Volume_20']
    
    # Calculate Price Breakout Indicator
    min_low_20 = df['low'].rolling(window=20).min()
    max_high_20 = df['high'].rolling(window=20).max()
    conditions = [df['close'] > max_high_20, df['close'] <= max_high_20]
    choices = [(df['close'] - min_low_20) / min_low_20 * 100, 0]
    df['PBI'] = pd.Series(pd.np.select(conditions, choices, default=0))
    
    # Calculate Open-Close Ratio and Number of Days with Positive OCR
    df['OCR'] = df['close'] / df['open']
    df['Positive_OCR_Count'] = (df['OCR'] > 1).rolling(window=20).sum()
    
    # Combine all factors to create the alpha factor
    alpha_factor = (df['ROC_5'] + df['ROC_20'] - df['Williams_R_14'] 
                    + df['VRS'] + df['PBI'] + df['Positive_OCR_Count'])
    
    return alpha_factor
