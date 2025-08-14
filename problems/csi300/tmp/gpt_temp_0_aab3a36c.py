import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Range
    df['intraday_range'] = (df['high'] - df['low']) / df['open']
    
    # Calculate Close-to-Open Return
    df['close_to_open_return'] = (df['close'] - df['open']) / df['open']
    
    # Create a new column for the alpha factor
    df['alpha_factor'] = 0.0
    
    # Dynamic Weighting Based on Recency
    recent_weights = {'intraday_range': 0.7, 'close_to_open_return': 0.3}
    older_weights = {'intraday_range': 0.5, 'close_to_open_return': 0.5}
    
    for i in range(len(df)):
        if i < 5:
            df.loc[df.index[i], 'alpha_factor'] = (
                recent_weights['intraday_range'] * df.loc[df.index[i], 'intraday_range'] +
                recent_weights['close_to_open_return'] * df.loc[df.index[i], 'close_to_open_return']
            )
        else:
            df.loc[df.index[i], 'alpha_factor'] = (
                older_weights['intraday_range'] * df.loc[df.index[i], 'intraday_range'] +
                older_weights['close_to_open_return'] * df.loc[df.index[i], 'close_to_open_return']
            )
    
    return df['alpha_factor']
