def heuristics_v2(df):
    avg_high_low = (df['high'].rolling(window=5).mean() + df['low'].rolling(window=5).mean()) / 2
    heuristics_matrix = (df['close'] / avg_high_low * np.log(df['volume'])).dropna()
    return heuristics_matrix
