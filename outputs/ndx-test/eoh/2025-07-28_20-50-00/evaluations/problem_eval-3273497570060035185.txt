def heuristics_v2(df):
    # Calculate the 21-day and 63-day cumulative returns
    cum_return_21 = (df['close'] / df['close'].shift(21)) - 1
    cum_return_63 = (df['close'] / df['close'].shift(63)) - 1
    # Compute the heuristic factor as the ratio of the 21-day to 63-day cumulative returns
    heuristics_matrix = cum_return_21 / cum_return_63.replace(0, pd.NA)
    return heuristics_matrix
