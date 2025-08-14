def heuristics_v2(df):
    # Calculate the high-low range
    high_low_range = df['high'] - df['low']
    
    # Calculate the ratio of the high-low range to the adjusted close price
    price_ratio = high_low_range / df['adj_close']
    
    # Apply a 21-day Exponential Moving Average (EMA) to the price ratio
    heuristics_matrix = price_ratio.ewm(span=21, adjust=False).mean()
    
    return heuristics_matrix
