def heuristics_v2(df):
    # Calculate the 50-day momentum
    df['Momentum_50'] = df['close'] - df['close'].shift(50)
    
    # Calculate the average daily trading volume over the last 10 days
    avg_volume_10 = df['volume'].rolling(window=10).mean()
    
    # Calculate the ratio of positive to negative returns over the past 20 days
    pos_returns = (df['close'].pct_change() > 0).rolling(window=20).sum()
    neg_returns = 20 - pos_returns
    pos_neg_ratio = pos_returns / neg_returns
    
    # Handle division by zero for the ratio
    pos_neg_ratio = pos_neg_ratio.replace([np.inf, -np.inf], np.nan).fillna(1)
    
    # Generate the heuristic matrix by combining the above factors
    heuristics_matrix = df['Momentum_50'] * avg_volume_10 * pos_neg_ratio
    
    return heuristics_matrix
