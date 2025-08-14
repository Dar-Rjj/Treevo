import pandas as pd
    
    # Calculate momentum of close prices
    momentum = df['close'].pct_change(periods=20)
    
    # Calculate the average volume over the last 20 days
    avg_volume = df['volume'].rolling(window=20).mean()
    
    # Compute the volume-to-average-volume ratio
    volume_ratio = df['volume'] / avg_volume
    
    # Combine momentum and volume ratio to form the heuristic
    heuristics_matrix = momentum * volume_ratio
    
    return pd.Series(heuristics_matrix, index=df.index)
```

This code snippet defines the `heuristics_v2` function, which calculates a new factor based on the specified logic and return heuristics_matrix
