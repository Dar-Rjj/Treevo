def heuristics_v2(df):
    weight = df['volume'] / df['volume'].sum()
    combined_prices_high_low = pd.concat([df['high'], df['low']])
    combined_weights = pd.concat([weight, weight]).reset_index(drop=True)
    weighted_median_price = (combined_prices_high_low * combined_weights).groupby(level=0).median()
    median_price = ((df['open'] + df['close']) / 2).median()
    heuristic_values = weighted_median_price - median_price
    heuristics_matrix = heuristic_values.ewm(span=7, adjust=False).mean().dropna()
    return heuristics_matrix
