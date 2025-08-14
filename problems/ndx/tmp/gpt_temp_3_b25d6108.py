import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate the rolling mean of close prices over a 10-day window
    df['close_10d_mean'] = df['close'].rolling(window=10).mean()
    
    # Calculate the rolling standard deviation of close prices over a 10-day window
    df['close_10d_std'] = df['close'].rolling(window=10).std()
    
    # Calculate the z-score of the close price
    df['close_z_score'] = (df['close'] - df['close_10d_mean']) / df['close_10d_std']
    
    # Calculate the ratio of high to low price
    df['high_low_ratio'] = df['high'] / df['low']
    
    # Calculate the log return from open to close
    df['log_return_open_close'] = (df['close'] / df['open']).apply(lambda x: math.log(x))
    
    # Combine the factors into a single alpha factor
    df['alpha_factor'] = df['close_z_score'] + df['high_low_ratio'] + df['log_return_open_close']
    
    # Return the alpha factor as a pandas Series
    return df['alpha_factor']
