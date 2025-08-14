import pandas as pd
    
    # Compute daily returns and their standard deviation
    returns = df['close'].pct_change()
    std_returns = returns.std()
    
    # Compute daily volume change in percentage and its standard deviation
    volume_change = df['volume'].pct_change()
    std_volume_change = volume_change.std()
    
    # Calculate the weights based on the inverse of their standard deviations
    weight_returns = 1 / std_returns
    weight_volume = 1 / std_volume_change
    
    # Generate the alpha factor using a weighted sum
    heuristics_matrix = (returns * weight_returns) + (volume_change * weight_volume)
    
    return heuristics_matrix
