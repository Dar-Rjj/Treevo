def heuristics_v2(df):
    # Calculate the 30-day rolling max and min of daily closing prices
    max_close = df['close'].rolling(window=30).max()
    min_close = df['close'].rolling(window=30).min()

    # Calculate the average of the daily trading volumes
    avg_volume = df['volume'].rolling(window=30).mean()

    # Compute the heuristic factor
    heuristic_factor = (max_close - min_close) / avg_volume if avg_volume.any() > 0 else 0

    # Create a Series for the heuristic factor, ensuring it is indexed correctly
    heuristics_matrix = pd.Series(heuristic_factor, index=df.index)
    return heuristics_matrix
