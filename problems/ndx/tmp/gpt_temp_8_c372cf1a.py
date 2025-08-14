def heuristics_v2(df):
    # Calculate short-term and long-term moving averages
    short_ma = df['close'].rolling(window=10).mean()
    long_ma = df['close'].rolling(window=50).mean()
    
    # Calculate the difference between short and long moving averages
    ma_diff = short_ma - long_ma
    
    # Incorporate the square of the volume to emphasize high volume days
    vol_squared = df['volume'] ** 2
    
    # Combine the moving average difference with the squared volume
    combined_factor = ma_diff * vol_squared
    
    # Rank the combined factor
    ranked_factor = combined_factor.rank(pct=True)
    
    return heuristics_matrix
