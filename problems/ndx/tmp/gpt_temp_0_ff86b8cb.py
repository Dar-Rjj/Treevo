def heuristics_v2(df):
    avg_price = (df['high'] + df['low']) / 2
    weight = df['volume'] / df['volume'].sum()
    volume_adjusted_median = (df[['open', 'close']] * weight).median(axis=1)
    heuristic_values = avg_price - volume_adjusted_median
    smoothed_values = heuristic_values.ewm(span=10, adjust=False).mean()
    volatility_insight = heuristic_values.rolling(window=5).std()
    heuristics_matrix = smoothed_values + volatility_insight
    return heuristics_matrix
