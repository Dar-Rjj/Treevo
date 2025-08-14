import pandas as pd
    import numpy as np
    
    # Calculate the average of high and low prices for each date
    avg_high_low = (df['high'] + df['low']) / 2
    
    # Calculate the ratio of closing price to this average
    ratio_close_avg = df['close'] / avg_high_low
    
    # Define the span for the EWM
    span = 10  # This can be tuned
    
    # Apply exponentially weighted function
    heuristics_matrix = ratio_close_avg.ewm(span=span, adjust=False).mean()
    
    return heuristics_matrix
