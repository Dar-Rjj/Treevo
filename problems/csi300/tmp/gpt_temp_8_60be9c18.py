import pandas as pd
    import numpy as np
    
    # Calculate moving averages and standard deviations over short and long term
    ma_short = df['close'].rolling(window=5).mean()
    ma_long = df['close'].rolling(window=20).mean()
    std_short = df['close'].rolling(window=5).std()
    std_long = df['close'].rolling(window=20).std()
    
    # Combine the factors with weights
    heuristics_matrix = (ma_short - ma_long) + (std_short - std_long)
    
    return heuristics_matrix
