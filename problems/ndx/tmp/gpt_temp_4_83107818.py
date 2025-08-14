def heuristics_v2(df):
    # Calculate the 90-day relative strength of closing prices
    relative_strength = df['close'] / df['close'].shift(90)
    
    # Calculate the True Range
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - df['close'].shift(1)).abs()
    tr3 = (df['low'] - df['close'].shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Adjust the True Range by the volume
    volume_adjusted_tr = true_range * (df['volume'] / df['volume'].rolling(window=30).mean())
    
    # Combine relative strength and adjusted True Range into a single factor
    combined_factor = relative_strength * volume_adjusted_tr
    
    # Apply a 15-day simple moving average to smooth the factor
    smoothed_factor = combined_factor.rolling(window=15).mean()
    
    heuristics_matrix = smoothed_factor.dropna()
    return heuristics_matrix
