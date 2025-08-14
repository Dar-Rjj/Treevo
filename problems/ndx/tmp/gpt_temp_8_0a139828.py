def heuristics_v2(df):
    avg_price = (df['high'] + df['low']) / 2
    weight = df['volume'] / df['volume'].sum()
    geo_mean = (df['open'] ** weight * df['close'] ** weight).cumprod() ** (1 / weight.sum())
    heuristic_values = np.log(avg_price) - np.log(geo_mean)
    smoothed_values = heuristic_values.ewm(span=21, adjust=False).mean()
    volatility_insight = heuristic_values.rolling(window=14).std()
    heuristics_matrix = smoothed_values + volatility_insight
    return heuristics_matrix
