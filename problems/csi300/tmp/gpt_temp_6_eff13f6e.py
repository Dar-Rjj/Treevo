import pandas as pd

def heuristics_v2(df):
    # Calculate short-term and long-term moving averages for High and Low
    short_high_avg = df['high'].rolling(window=7).mean()
    long_high_avg = df['high'].rolling(window=21).mean()
    short_low_avg = df['low'].rolling(window=7).mean()
    long_low_avg = df['low'].rolling(window=21).mean()

    # Compute the difference between short-term and long-term averages
    high_diff = short_high_avg - long_high_avg
    low_diff = short_low_avg - long_low_avg

    # Calculate the log change in volume
    log_volume_change = df['volume'].apply(lambda x: x if x == 0 else (x+1)).pct_change().apply(lambda x: x if pd.isna(x) or x == 0 else (x+1)).apply(np.log)

    # Generate the heuristics matrix
    heuristics_matrix = (high_diff - low_diff) + log_volume_change
    
    return heuristics_matrix
