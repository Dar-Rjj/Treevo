import pandas as pd
    
    # Calculate the rolling mean of high and low prices
    rolling_high = df['high'].rolling(window=20).mean()
    rolling_low = df['low'].rolling(window=20).mean()
    
    # Calculate the difference between the rolling means of high and low
    hl_diff = rolling_high - rolling_low
    
    # Calculate the volume-weighted price change
    vwp_change = (df['close'] - df['open']) * df['volume']
    
    # Generate the heuristics matrix
    heuristics_matrix = (hl_diff + vwp_change) / 2
    
    return heuristics_matrix
