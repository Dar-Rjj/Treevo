def heuristics_v2(df):
    high, low, close, open_ = df['high'], df['low'], df['close'], df['open']
    
    overnight_return = (open_ - close.shift(1)) / close.shift(1)
    morning_range = (high - open_) / open_
    range_efficiency = (close - open_) / morning_range
    
    heuristics_matrix = -overnight_return * range_efficiency
    heuristics_matrix = heuristics_matrix.replace([float('inf'), -float('inf')], float('nan'))
    
    return heuristics_matrix
