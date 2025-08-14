import pandas as pd
    
    # Calculate the 5-day rate of change
    df['roc_5'] = df['close'].pct_change(5)
    
    # Calculate the 20-day simple moving average
    df['sma_20'] = df['close'].rolling(window=20).mean()
    
    # Calculate the difference between close price and its 20-day SMA
    df['diff_close_sma_20'] = df['close'] - df['sma_20']
    
    # Apply a weighted sum formula combining roc_5 and diff_close_sma_20
    df['heuristics_matrix'] = 0.6 * df['roc_5'] + 0.4 * df['diff_close_sma_20']
    
    return heuristics_matrix
