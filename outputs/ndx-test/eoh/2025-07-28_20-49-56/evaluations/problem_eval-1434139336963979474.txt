def heuristics_v2(df):
    # Calculate 12-day and 26-day Exponential Moving Averages (EMA)
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    
    # Determine the trend direction
    trend_direction = (ema_12 - ema_26) / df['close']
    
    # Calculate On-Balance Volume (OBV)
    obv = (np.sign(df['close'].diff()) * df['volume']).cumsum()
    
    # Assign dynamic weights based on the trend direction
    weight_trend = np.where(trend_direction > 0, 0.7, 0.3)
    weight_obv = 1 - weight_trend
    
    # Combine EMA crossover and OBV into a single heuristic measure
    heuristics_matrix = (weight_trend * trend_direction + weight_obv * (obv.diff() / df['volume']))
    
    return heuristics_matrix
