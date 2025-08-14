def heuristics_v2(df):
    high_low_diff = df['high'] - df['low']
    log_transform = np.log(high_low_diff)
    heuristics_matrix = log_transform.ewm(span=14, adjust=False).mean().dropna()
    
    return heuristics_matrix
