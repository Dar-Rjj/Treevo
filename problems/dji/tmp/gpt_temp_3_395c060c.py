import pandas as pd
    
    # Momentum factors
    df['mom_10'] = df['close'].pct_change(10)
    df['mom_30'] = df['close'].pct_change(30)
    
    # Volatility factors
    df['vol_10'] = df['close'].rolling(window=10).std()
    df['vol_30'] = df['close'].rolling(window=30).std()
    
    # Volume trend
    df['volume_change'] = df['volume'].pct_change()
    
    # Composite heuristic
    df['composite'] = (df['mom_10'] + df['mom_30']) / 2 - (df['vol_10'] + df['vol_30']) / 2 + df['volume_change']
    
    # Drop NA values
    df.dropna(inplace=True)
    
    heuristics_matrix = df['composite']
    
    return heuristics_matrix
