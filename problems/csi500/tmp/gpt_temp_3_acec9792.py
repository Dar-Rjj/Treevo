import pandas as pd
import numpy as np
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Price Movement
    df['intraday_movement'] = (df['high'] - df['low']) / df['low']
    
    # Calculate Previous Day's Range
    df['prev_day_range'] = (df['high'].shift(1) - df['low'].shift(1)) / df['low'].shift(1)
    
    # Determine if Current Day's Movement is Greater than Previous Day's Range
    df['movement_greater'] = df['intraday_movement'] > df['prev_day_range']
    
    # Analyze Volume Changes
    df['volume_increase'] = df['volume'] > df['volume'].shift(1)
    
    # Combine Momentum and Volume Indicators
    conditions = [
        (df['intraday_movement'] > 0) & (df['movement_greater']) & (df['volume_increase']),
        (df['intraday_movement'] > 0) & (df['movement_greater']) & (~df['volume_increase']),
        (df['intraday_movement'] <= 0) | (~df['movement_greater']) & (df['volume_increase']),
        (df['intraday_movement'] <= 0) | (~df['movement_greater']) & (~df['volume_increase'])
    ]
    choices = [3, 2, -3, -2]  # Strong Buy, Weak Buy, Strong Sell, Weak Sell
    
    df['signal'] = pd.np.select(conditions, choices, default=0)
    
    return df['signal']
