import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Range
    intraday_range = df['high'] - df['low']
    
    # Calculate Close-to-Open Return
    close_to_open_return = (df['close'] - df['open']) / df['open']
    
    # Define weights based on recency
    recent_weights = {'intraday_range': 0.7, 'close_to_open_return': 0.3}
    older_weights = {'intraday_range': 0.5, 'close_to_open_return': 0.5}
    
    # Incorporate Volume
    def adjust_weights(volume):
        if volume > df['volume'].mean():
            return {'intraday_range': 0.8, 'close_to_open_return': 0.2}
        else:
            return {'intraday_range': 0.4, 'close_to_open_return': 0.6}
    
    # Combine Intraday Range and Close-to-Open Return with adjusted weights
    df['alpha_factor'] = 0
    for i in range(1, len(df)):
        weights = recent_weights if i < len(df) // 2 else older_weights
        volume_adjusted_weights = adjust_weights(df['volume'].iloc[i])
        df.loc[df.index[i], 'alpha_factor'] = (
            volume_adjusted_weights['intraday_range'] * intraday_range.iloc[i] +
            volume_adjusted_weights['close_to_open_return'] * close_to_open_return.iloc[i]
        )
    
    return df['alpha_factor']
