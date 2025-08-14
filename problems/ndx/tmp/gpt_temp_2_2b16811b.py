import pandas as pd

def heuristics_v2(df):
    # Calculate the 14-day and 28-day exponential moving averages of the close price
    ema_14 = df['close'].ewm(span=14, adjust=False).mean()
    ema_28 = df['close'].ewm(span=28, adjust=False).mean()
    
    # Calculate the ratio between the EMAs
    ema_ratio = ema_14 / ema_28
    
    # Calculate the 10-day sum of log volume
    log_volume = df['volume'].apply(lambda x: 0 if x <= 0 else np.log(x))
    sum_log_volume_10 = log_volume.rolling(window=10).sum()
    
    # Generate the heuristic matrix by adding the EMA ratio to the sum of log volume
    heuristics_matrix = ema_ratio + sum_log_volume_10
    
    return heuristics_matrix
