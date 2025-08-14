def heuristics_v2(df):
    # Calculate the difference between close and open prices
    price_diff = df['close'] - df['open']
    
    # Calculate the 30-day EWMA of the price difference
    ewma_price_diff = price_diff.ewm(span=30, adjust=False).mean()
    
    return heuristics_matrix
