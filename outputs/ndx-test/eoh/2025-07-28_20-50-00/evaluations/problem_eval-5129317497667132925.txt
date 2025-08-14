import pandas as pd
    
    # Calculate price change and volume change
    df['price_change'] = df['close'].pct_change()
    df['volume_change'] = df['volume'].pct_change()
    
    # Initialize weights for each feature
    weights = pd.Series(1, index=['open', 'high', 'low', 'close', 'volume'])
    
    # Adjust weights based on recent performance (last 5 days)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        weights[col] *= (df[col].pct_change().rolling(window=5).mean().shift(1) + 1)
    
    # Calculate the weighted sum of market features
    heuristics_matrix = (df[['open', 'high', 'low', 'close', 'volume']] * weights).sum(axis=1)
    
    # Apply a lag to capture lead-lag relationship
    heuristics_matrix = heuristics_matrix.shift(1)
    
    return heuristics_matrix
