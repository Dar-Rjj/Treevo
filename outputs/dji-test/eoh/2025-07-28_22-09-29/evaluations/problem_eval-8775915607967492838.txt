import pandas as pd
    
    # Calculate short, medium, and long-term momentum
    short_momentum = df['close'].pct_change(periods=5)
    medium_momentum = df['close'].pct_change(periods=20)
    long_momentum = df['close'].pct_change(periods=60)
    
    # Volume trend over the long term
    volume_trend = df['volume'].pct_change(periods=60)
    
    # Combine all factors into a single heuristic
    heuristics_matrix = 0.4*short_momentum + 0.3*medium_momentum + 0.2*long_momentum + 0.1*volume_trend
    
    return heuristics_matrix
