def heuristics_v2(df):
    # Calculate the numerator as the difference between close and open prices
    numerator = df['close'] - df['open']
    # Calculate the denominator as the average of high and low prices
    denominator = (df['high'] + df['low']) / 2
    # Generate the raw heuristic factor
    raw_heuristic = (numerator / denominator) * df['volume']
    # Return the final heuristics as a pandas Series
    return heuristics_matrix
