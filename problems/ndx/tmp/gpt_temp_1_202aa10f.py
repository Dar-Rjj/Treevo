import pandas as pd

def heuristics_v2(df):
    # Calculate the 14-day and 60-day exponential moving averages of the close price
    ema_14 = df['close'].ewm(span=14, adjust=False).mean()
    ema_60 = df['close'].ewm(span=60, adjust=False).mean()
    
    # Calculate the ratio between the EMAs
    ema_ratio = ema_14 / ema_60
    
    # Calculate the daily logarithmic return
    df['Log_Return'] = df['close'].pct_change().apply(lambda x: np.log(1 + x))
    
    # Calculate the 40-day standard deviation of daily logarithmic returns
    sd_40 = df['Log_Return'].rolling(window=40).std()
    
    # Generate the heuristic matrix by multiplying the EMA ratio with the SD
    heuristics_matrix = ema_ratio * sd_40
    
    return heuristics_matrix
