import pandas as pd
    import numpy as np
    
    # Calculate momentum
    df['Momentum'] = df['close'].pct_change(periods=1)
    
    # Calculate RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Composite factor
    df['CompositeFactor'] = df['Momentum'] * df['RSI']
    
    heuristics_matrix = df['CompositeFactor'].copy()

    return heuristics_matrix
