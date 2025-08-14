import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Range
    intraday_range = df['high'] - df['low']
    
    # Calculate Close-to-Open Return
    close_to_open_return = (df['close'] - df['open'].shift(1)) / df['open'].shift(1)
    
    # Compute Intraday Volatility
    intraday_volatility = (df['high'] - df['low']) / (df['high'] + df['low'])
    
    # Determine the weights based on recent or older data
    recent_weights = 0.7 * intraday_volatility + 0.3 * close_to_open_return
    older_weights = 0.5 * intraday_volatility + 0.5 * close_to_open_return
    
    # Combine Intraday Volatility and Close-to-Open Return with dynamic weighting
    factor_values = recent_weights.where(df.index > df.index[-1] - pd.Timedelta(days=30), 
                                         other=older_weights)
    
    return factor_values
