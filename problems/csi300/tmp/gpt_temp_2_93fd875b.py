import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Range
    intraday_range = df['high'] - df['low']
    
    # Calculate Close-to-Open Return
    close_to_open_return = (df['close'].shift(-1) - df['open']) / df['open']
    
    # Assign weights based on recency
    recent_data = df.index > df.index[-1] - pd.Timedelta(days=30)
    older_data = ~recent_data
    
    # Combine Intraday Range and Close-to-Open Return with dynamic weighting
    factor = pd.Series(index=df.index, dtype=float)
    factor[recent_data] = 0.7 * intraday_range[recent_data] + 0.3 * close_to_open_return[recent_data]
    factor[older_data] = 0.5 * intraday_range[older_data] + 0.5 * close_to_open_return[older_data]
    
    return factor
