import pandas as pd
    # Calculate the 20-day rolling mean of the 'Close' price
    df['rolling_mean_close'] = df['Close'].rolling(window=20).mean()
    # Find the deviation of the 'Close' price from its 20-day rolling mean
    df['deviation'] = df['Close'] - df['rolling_mean_close']
    # Create a simple weight factor based on the ratio of current volume to its 5-day mean
    df['volume_weight'] = df['Volume'] / df['Volume'].rolling(window=5).mean()
    # Generate the heuristic value by multiplying the deviation with the volume weight
    heuristics_matrix = (df['deviation'] * df['volume_weight']).dropna()
    return heuristics_matrix
