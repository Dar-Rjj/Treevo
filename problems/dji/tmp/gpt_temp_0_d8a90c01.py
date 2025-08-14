def heuristics_v2(df):
    # Calculate Daily Price Change
    df['price_change'] = df['close'] - df['close'].shift(1)
    
    # Compute Volume-Weighted Daily Return
    N = 20
    df['volume_weighted_return'] = (df['price_change'] * df['volume']).rolling(window=N).sum() / df['volume'].rolling(window=N).sum()
