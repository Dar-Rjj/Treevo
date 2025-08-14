def heuristics_v2(df, window):
    # Calculate Price Impulse
    price_impulse = df['close'].diff()
    
    # Determine Volume Trend Direction
    volume_trend_direction = (df['volume'] > df['volume'].shift(1)).astype(int)
    
    # Combine Price and Volume Indicators
    combined_indicator = price_impulse * (volume_trend_direction * 2 + 1)
    
    # Sum Over a Window
    factor_values = combined_indicator.rolling(window=window).sum()
    
    return factor_values
