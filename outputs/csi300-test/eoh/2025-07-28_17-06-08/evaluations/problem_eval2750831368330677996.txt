import pandas as pd

def heuristics_v4(df):
    close_ema = df['close'].ewm(span=20, adjust=False).mean()
    range_change = (df['high'] - df['low']).pct_change()
    volume_ema = df['volume'].ewm(span=7, adjust=False).mean()
    
    # Compute historical correlation for weighting (simplified using static values here for example)
    corr_close_return = 0.6  # Example value, to be computed dynamically
    corr_range_change = 0.3  # Example value, to be computed dynamically
    corr_volume_ema = 0.1    # Example value, to be computed dynamically
    
    heuristics_matrix = (corr_close_return * close_ema + 
                         corr_range_change * range_change + 
                         corr_volume_ema * volume_ema).dropna()
    return heuristics_matrix
