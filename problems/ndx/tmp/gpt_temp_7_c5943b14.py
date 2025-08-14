def heuristics_v2(df):
    # Calculate daily return
    df['daily_return'] = df['close'].pct_change()
    
    # Heuristic 1: Volume weighted by daily return
    h1 = df['volume'] * df['daily_return'].shift(1)
    
    # Heuristic 2: Difference between today's high and yesterday's close, adjusted by volume
    h2 = (df['high'] - df['close'].shift(1)) / df['close'].shift(1) * df['volume']
    
    # Heuristic 3: Ratio of volume change to price change
    h3 = (df['volume'] - df['volume'].shift(1)) / (df['close'] - df['close'].shift(1))
    
    # Combine into a single DataFrame
    heuristics_matrix = pd.DataFrame({'h1': h1, 'h2': h2, 'h3': h3})
    
    # Return the combined heuristics as a Series with multi-level index for date and heuristic type
    return heuristics_matrix
