import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Range
    df['intraday_range'] = df['high'] - df['low']
    
    # Calculate Close-to-Open Return
    df['close_to_open_return'] = (df['close'] - df['open'].shift(1)) / df['open'].shift(1)
    
    # Assign dynamic weights to recent and older data
    def weighted_combination(row, recent_weight, older_weight):
        if row.name < 0.5 * len(df):  # Define "older" as the first half of the data
            return older_weight * row['intraday_range'] + (1 - older_weight) * row['close_to_open_return']
        else:  # Define "recent" as the second half of the data
            return recent_weight * row['intraday_range'] + (1 - recent_weight) * row['close_to_open_return']
    
    # Apply the weighted combination
    df['alpha_factor'] = df.apply(lambda row: weighted_combination(row, 0.7, 0.5), axis=1)
    
    return df['alpha_factor']
