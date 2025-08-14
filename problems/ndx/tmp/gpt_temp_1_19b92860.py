import pandas as pd

def heuristics_v2(df):
    # Calculate the 20-day moving average of the log of volume
    ma_log_volume = df['volume'].apply(lambda x: max(1, x)).apply(np.log).rolling(window=20).mean()
    
    # Calculate the 20-day exponential moving average of close price
    ema_close = df['close'].ewm(span=20, adjust=False).mean()
    
    # Generate the heuristic factor by multiplying the MA log volume with the EMA of close
    heuristics_matrix = ma_log_volume * ema_close
    
    return heuristics_matrix
