def heuristics_v2(df):
    # Calculate the ADX for trend strength
    adx = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
    
    # Calculate the relative strength factor as the ratio of closing price to open price
    rs_factor = df['close'] / df['open']
    
    # Apply a custom heuristic to combine the ADX and the relative strength factor
    heuristics_matrix = (adx + rs_factor).rank(pct=True)
    
    return heuristics_matrix
