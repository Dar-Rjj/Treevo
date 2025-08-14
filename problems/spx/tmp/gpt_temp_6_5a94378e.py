def heuristics_v2(df):
    # Calculate Volume-Weighted Price (VWP)
    df['VWP'] = df['close'] * df['volume']
    
    # Calculate Daily Returns
    df['daily_returns'] = df['close'].pct_change()
