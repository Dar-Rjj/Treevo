def heuristics_v2(df):
    # Calculate simple moving averages for close prices over 5 and 20 days
    df['SMA_5'] = df['close'].rolling(window=5).mean()
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    
    # Calculate the difference between the 5-day and 20-day SMAs
    df['SMA_diff'] = df['SMA_5'] - df['SMA_20']
    
    # Calculate the momentum of the close prices as the change over 10 days
    df['momentum'] = df['close'].pct_change(periods=10)
    
    # Generate the alpha factor: combine SMA difference and momentum
    df['alpha_factor'] = (df['SMA_diff'] + df['momentum']) * (df['volume'].pct_change())
    
    # Return only the alpha_factor column as a Series
    return heuristics_matrix
