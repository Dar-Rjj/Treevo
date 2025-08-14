def heuristics_v2(df):
    # Calculate Daily Price Movement Range
    df['price_range'] = df['high'] - df['low']
    
    # Determine Daily Return Deviation from Close
    df['daily_return_deviation'] = df['close'].diff()
