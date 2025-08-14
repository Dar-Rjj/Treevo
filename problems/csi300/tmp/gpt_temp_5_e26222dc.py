import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Range
    df['intraday_range'] = df['high'] - df['low']
    
    # Calculate Close-to-Open Return
    df['close_to_open_return'] = (df['close'].shift(-1) - df['close']) / df['close']
    
    # Determine recent and older weights
    def apply_weights(row, recent_threshold=5):
        if row.name < recent_threshold:
            return 0.7 * row['intraday_range'] + 0.3 * row['close_to_open_return']
        else:
            return 0.5 * row['intraday_range'] + 0.5 * row['close_to_open_return']
    
    # Apply the dynamic weighting
    df['alpha_factor'] = df.apply(apply_weights, axis=1)
    
    # Return the alpha factor as a pandas Series
    return df['alpha_factor']
