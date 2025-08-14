import pandas as pd

def heuristics_v2(df):
    # Calculate the difference between high and low for each day
    diff_high_low = df['high'] - df['low']
    # Compute the ratio of this difference to the volume
    ratio = diff_high_low / df['volume']
    # Calculate the exponential moving average of this ratio, using a 10-day span as an example
    ema_ratio = ratio.ewm(span=10, adjust=False).mean()
    # Return the EMA series as the heuristics_matrix
    return heuristics_matrix
