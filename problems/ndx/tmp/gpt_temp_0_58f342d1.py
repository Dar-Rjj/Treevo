def heuristics_v2(df):
    # Calculate the difference between close and open prices
    co_diff = df['close'] - df['open']
    
    # Calculate the cumulative sum of the difference
    cum_sum_diff = co_diff.cumsum()
    
    # Calculate the square root of the sum of squared volumes
    sqrt_vol = np.sqrt((df['volume'] ** 2).cumsum())
    
    # Compute the factor by dividing the cumulative sum of differences by the square root of the sum of squared volumes
    factor = cum_sum_diff / sqrt_vol
    
    # Apply a rolling window standard deviation
    std_factor = factor.rolling(window=20).std()
    
    # Rank the resulting factors
    heuristics_matrix = std_factor.rank(pct=True)
    
    return heuristics_matrix
