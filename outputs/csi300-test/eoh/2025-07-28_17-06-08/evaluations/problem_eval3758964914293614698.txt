def heuristics_v2(df):
    # Calculate the weighted moving average of volume
    df['wma_volume'] = df['volume'].rolling(window=20).apply(lambda x: (x * range(1, len(x) + 1)).sum() / sum(range(1, len(x) + 1)), raw=True)
    
    # Calculate the difference between close and the WMA of volume
    df['heuristic'] = df['close'] - df['wma_volume']
    
    # Generate a trend-following signal
    df['signal'] = df['heuristic'].pct_change().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    
    # Combine the heuristic with the signal
    df['heuristics_matrix'] = df['heuristic'] * df['signal']
    
    return heuristics_matrix
