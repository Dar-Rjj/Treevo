def heuristics_v2(df):
    log_return = np.log(df['close']).diff(5)
    volume_sma = df['volume'].rolling(window=20).mean()
    heuristic_values = log_return - volume_sma
    heuristics_matrix = heuristic_values.ewm(span=7, adjust=False).mean().dropna()
    return heuristics_matrix
