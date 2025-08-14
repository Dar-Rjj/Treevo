def heuristics_v2(df):
    # Calculate 5 and 20 day Moving Averages
    ma_5 = df['close'].rolling(window=5).mean()
    ma_20 = df['close'].rolling(window=20).mean()
    
    # Compute the difference between High and Low prices
    diff_high_low = df['high'] - df['low']
    
    # Adjust the difference by the natural logarithm of volume
    adj_diff_high_low = diff_high_low * (1 + np.log(df['volume']))
    
    # Construct the heuristic factor
    heuristics_matrix = (ma_5 - ma_20) + adj_diff_high_low
    
    return heuristics_matrix
