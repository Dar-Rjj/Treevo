def heuristics_v2(df):
    log_return = np.log(df['close'] / df['close'].shift(1))
    roc_volume = (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1)
    factor = log_return * roc_volume
    heuristics_matrix = factor.ewm(span=20, adjust=False).mean().dropna()
    
    return heuristics_matrix
