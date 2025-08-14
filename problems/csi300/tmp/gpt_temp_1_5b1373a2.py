import pandas as pd
import numpy as np
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Range
    df['intraday_range'] = df['high'] - df['low']
    
    # Calculate Close-to-Open Return
    df['close_to_open_return'] = (df['close'] - df['open'].shift(1)) / df['open'].shift(1)
    
    # Assess Recency
    df['is_recent'] = df.index.isin(df.index[-5:])
    
    # Assess Volume
    median_volume = df['volume'].median()
    df['is_high_volume'] = df['volume'] > median_volume
    
    # Define weights based on recency and volume
    conditions = [
        (df['is_recent'] & df['is_high_volume']),
        (df['is_recent'] & ~df['is_high_volume']),
        (~df['is_recent'] & df['is_high_volume']),
        (~df['is_recent'] & ~df['is_high_volume'])
    ]
    
    choices = [
        0.8 * df['intraday_range'] + 0.2 * df['close_to_open_return'],
        0.7 * df['intraday_range'] + 0.3 * df['close_to_open_return'],
        0.6 * df['intraday_range'] + 0.4 * df['close_to_open_return'],
        0.5 * df['intraday_range'] + 0.5 * df['close_to_open_return']
    ]
    
    # Apply the conditions to create the final alpha factor
    df['alpha_factor'] = pd.np.select(conditions, choices, default=0)
    
    return df['alpha_factor']
