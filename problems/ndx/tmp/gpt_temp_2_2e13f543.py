import pandas as pd
    
    # Calculate daily returns
    df['returns'] = df['close'].pct_change()
    
    # Compute rolling correlation between daily returns and volume
    df['corr_returns_volume'] = df['returns'].rolling(window=30).corr(df['volume'])
    
    # Generate the heuristics matrix by averaging the correlations over a specified window
    heuristics_matrix = df['corr_returns_volume'].rolling(window=60).mean().dropna()
    
    return heuristics_matrix
