import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Momentum
    intraday_momentum = (df['high'] - df['low']) / df['open']
    
    # Calculate Volume Relative Change
    volume_relative_change = (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1)
    
    # Combine Intraday Momentum and Volume Change
    combined_factor_1 = intraday_momentum * volume_relative_change
    combined_factor_2 = intraday_momentum + volume_relative_change
    
    # Return the combined factor
    return combined_factor_1, combined_factor_2
