import pandas as pd
    
    # Calculate EMAs for two different spans
    ema_short = df['close'].ewm(span=10, adjust=False).mean()
    ema_long = df['close'].ewm(span=30, adjust=False).mean()

    # Generate the heuristic matrix by computing the difference between the short and long EMAs
    heuristics_matrix = ema_short - ema_long
    
    return heuristics_matrix
