def heuristics_v2(df):
    # Calculate the 20-day moving average of daily high prices
    ma_high_20 = df['high'].rolling(window=20).mean()
    
    # Calculate the 15-day moving average of daily low prices
    ma_low_15 = df['low'].rolling(window=15).mean()
    
    # Compute the heuristic factor as the ratio of these two moving averages
    heuristic_factor = ma_high_20 / ma_low_15
    
    # Create a Series for the heuristic factor, ensuring it is indexed correctly
    heuristics_matrix = pd.Series(heuristic_factor, index=df.index)
    return heuristics_matrix
