import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Range
    df['intraday_range'] = df['high'] - df['low']
    
    # Calculate Close-to-Open Return
    df['close_to_open_return'] = (df['close'] - df['open']) / df['open']
    
    # Shift the close price to get the previous day's close
    df['prev_close'] = df['close'].shift(1)
    
    # Initialize the factor values
    df['factor_value'] = 0.0
    
    # Define initial weights
    recent_intraday_weight = 0.7
    recent_close_open_weight = 0.3
    older_intraday_weight = 0.5
    older_close_open_weight = 0.5
    
    for i in range(1, len(df)):
        # Evaluate Market Dynamics
        if df['volume'].iloc[i] > df['volume'].iloc[i-1]:
            # Increase weight for Intraday Range and decrease for Close-to-Open Return
            intraday_weight = recent_intraday_weight + 0.1
            close_open_weight = recent_close_open_weight - 0.1
        else:
            # Decrease weight for Intraday Range and increase for Close-to-Open Return
            intraday_weight = recent_intraday_weight - 0.1
            close_open_weight = recent_close_open_weight + 0.1
        
        # Ensure weights are within [0, 1]
        intraday_weight = max(0, min(1, intraday_weight))
        close_open_weight = max(0, min(1, close_open_weight))
        
        # Combine Intraday Range and Close-to-Open Return with adjusted weights
        df.loc[df.index[i], 'factor_value'] = (intraday_weight * df['intraday_range'].iloc[i] +
                                              close_open_weight * df['close_to_open_return'].iloc[i])
    
    return df['factor_value']
