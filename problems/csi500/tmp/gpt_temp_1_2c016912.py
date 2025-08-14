import pandas as pd
import numpy as np
import pandas as pd

def heuristics(df):
    # Calculate the 5-day and 20-day EMA
    df['5_day_EMA'] = df['close'].ewm(span=5, adjust=False).mean()
    df['20_day_EMA'] = df['close'].ewm(span=20, adjust=False).mean()
    
    # Compute the EMA Crossover Signal
    df['EMA_Crossover_Signal'] = (df['5_day_EMA'] > df['20_day_EMA']).astype(int)
    
    # Calculate the True Range for each day
    df['true_range'] = df[['high', 'low', 'close']].max(axis=1) - df[['high', 'low', 'close']].min(axis=1)
    
    # Calculate the 14-day ATR
    df['14_day_ATR'] = df['true_range'].rolling(window=14).mean()
    
    # Define Upper and Lower Volatility Bands
    multiplier = 2.0
    df['Upper_Band'] = df['close'] + multiplier * df['14_day_ATR']
    df['Lower_Band'] = df['close'] - multiplier * df['14_day_ATR']
    
    # Determine Position Relative to Volatility Bands
    conditions = [
        (df['close'] > df['Upper_Band']),
        (df['close'] < df['Lower_Band']),
        (df['close'] <= df['Upper_Band']) & (df['close'] >= df['Lower_Band'])
    ]
    choices = [1, -1, 0]
    df['Position'] = pd.np.select(conditions, choices, default=0)
    
    # Combine Factors into a Composite Score
    df['Composite_Score'] = 0
    high_ATR_threshold = df['14_day_ATR'].quantile(0.75)
    
    # Condition 1: Strong momentum with high volatility and price breakout
    df.loc[(df['EMA_Crossover_Signal'] == 1) & (df['14_day_ATR'] > high_ATR_threshold) & (df['Position'] == 1), 'Composite_Score'] = 5
    # Condition 2: Strong momentum with high volatility but within bands
    df.loc[(df['EMA_Crossover_Signal'] == 1) & (df['14_day_ATR'] > high_ATR_threshold) & (df['Position'] == 0), 'Composite_Score'] = 4
    # Condition 3: Strong momentum with low volatility and within bands
    df.loc[(df['EMA_Crossover_Signal'] == 1) & (df['14_day_ATR'] <= high_ATR_threshold) & (df['Position'] == 0), 'Composite_Score'] = 3
    # Condition 4: No clear upward trend
    df.loc[df['EMA_Crossover_Signal'] == 0, 'Composite_Score'] = 1
    
    return df['Composite_Score']
