import pandas as pd
import pandas as pd

def heuristics_v2(df, volume_threshold):
    # Calculate Intraday High-to-Low Range
    intraday_range = df['high'] - df['low']
    
    # Calculate Close-to-Previous Close Return
    close_to_prev_close_return = df['close'].pct_change().fillna(0)
    
    # Combine Indicators
    combined_indicator = intraday_range * close_to_prev_close_return
    
    # Apply Volume Filter
    factor_values = combined_indicator.where(df['volume'] > volume_threshold, 0)
    
    return factor_values
