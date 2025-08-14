import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Range
    intraday_range = df['high'] - df['low']
    
    # Calculate Close-to-Open Return
    close_to_open_return = (df['close'] - df['open']) / df['open']

    # Compute Adaptive Exponential Weighting
    recent_weights = [0.7, 0.3]  # Weights for Intraday Range and Close-to-Open Return
    older_weights = [0.5, 0.5]
    
    # Identify Volume Shocks
    avg_volume = df['volume'].rolling(window=30).mean()
    volume_shock = df['volume'] > (avg_volume * 1.5)
    
    # Adjust Weights Based on Volume Shocks
    weights = np.where(volume_shock[:, None], recent_weights, older_weights)
    
    # Combine Intraday Range and Close-to-Open Return
    combined_factor = (weights[0] * intraday_range) + (weights[1] * close_to_open_return)
    
    return pd.Series(combined_factor, index=df.index, name='alpha_factor')
