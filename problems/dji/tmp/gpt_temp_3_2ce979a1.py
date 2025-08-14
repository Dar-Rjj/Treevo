def heuristics_v2(df):
    # Calculate Short-Term Return
    df['short_term_return'] = (df['close'] / df['close'].shift(5) - 1)
