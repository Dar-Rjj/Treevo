def heuristics_v2(df):
    vol_change = df['volume'].pct_change(periods=10)
    smoothed_vol_change = vol_change.rolling(window=30).mean().dropna()
    return heuristics_matrix
