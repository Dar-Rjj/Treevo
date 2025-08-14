def heuristics_v2(df):
    # Calculate Volume-Weighted Price
    df['vw_price'] = df['volume'] * df['close']
    
    # Calculate Daily Return using volume-weighted price
    df['daily_return'] = df['vw_price'] / df['vw_price'].shift(1) - 1
