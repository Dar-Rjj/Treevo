import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df, n=20):
    # Calculate Intraday Range
    df['Intraday_Range'] = df['high'] - df['low']
    
    # Calculate Close-to-Open Return
    df['Prev_Close'] = df['close'].shift(1)
    df['Close_to_Open_Return'] = (df['close'] - df['open']) / df['open']
    
    # Incorporate Volume Shock
    df['Vol_Moving_Avg'] = df['volume'].rolling(window=n).mean().shift(1)
    df['Volume_Shock'] = (df['volume'] - df['Vol_Moving_Avg']) / df['Vol_Moving_Avg']
    
    # Assign dynamic weights based on the age of the data
    recent_weights = np.array([0.5, 0.3, 0.2])
    older_weights = np.array([0.4, 0.4, 0.2])
    
    def combine_factors(row):
        if row.name < n:
            return np.nan
        else:
            weights = recent_weights if (row.name >= len(df) - n) else older_weights
            return np.dot(weights, [row['Intraday_Range'], row['Close_to_Open_Return'], row['Volume_Shock']])
    
    # Combine Intraday Range, Close-to-Open Return, and Volume Shock
    df['Factor_Value'] = df.apply(combine_factors, axis=1)
    
    # Apply Exponential Weighting
    df['Exponentially_Weighted_Factor'] = df['Factor_Value'].ewm(alpha=1-0.95, adjust=False).mean()
    
    return df['Exponentially_Weighted_Factor']
