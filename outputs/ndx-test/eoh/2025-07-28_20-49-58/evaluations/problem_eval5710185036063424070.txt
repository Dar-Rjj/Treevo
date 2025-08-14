import pandas as pd
    import numpy as np
    
    # Calculate the 30-day weighted sum of close prices' changes
    close_changes = df['close'].diff()
    weights = np.array([0.95 ** i for i in range(30)])
    weighted_sum = close_changes.rolling(window=30).apply(lambda x: (x * weights[:len(x)]).sum(), raw=False)
    
    # Calculate 5-day moving average of volumes
    volume_avg_5 = df['volume'].rolling(window=5).mean()
    
    # Volume ratio
    volume_ratio = df['volume'] / volume_avg_5
    
    # Combine factors
    heuristics_matrix = weighted_sum + volume_ratio
    
    return heuristics_matrix
