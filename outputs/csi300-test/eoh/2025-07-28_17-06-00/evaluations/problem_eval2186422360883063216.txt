def heuristics_v2(df):
    # Calculate daily price change
    price_change = df['close'].diff(1).fillna(0)
    
    # Calculate the difference between high and low as a measure of intraday volatility, then smooth it using a longer rolling window
    intraday_volatility = (df['high'] - df['low']).rolling(window=20).mean()
    
    # Adjust the cumulative volume difference by giving more weight to days with higher positive price changes
    adjusted_volume_diff = (df['volume'] * np.where(price_change > 0, price_change, 0) - df['volume'] * np.where(price_change < 0, price_change, 0)).cumsum()
    
    # Combine the smoothed intraday volatility with the adjusted volume difference
    heuristics_matrix = intraday_volatility * adjusted_volume_diff
    
    return heuristics_matrix
