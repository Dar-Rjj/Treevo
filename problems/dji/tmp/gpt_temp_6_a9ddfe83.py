def heuristics_v2(df):
    hl_diff = df['high'] - df['low']
    sqrt_volume = df['volume'].apply(np.sqrt)
    heuristics_matrix = (hl_diff * sqrt_volume).rolling(window=7).mean().dropna()
    return heuristics_matrix
