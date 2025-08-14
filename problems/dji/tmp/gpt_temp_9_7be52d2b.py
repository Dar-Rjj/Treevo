def heuristics_v2(df):
    ema_10 = df['close'].ewm(span=10, adjust=False).mean()
    heuristics_matrix = ((df['close'].shift(-1) / ema_10) * np.log(df['volume'] + 1)).dropna()
    return heuristics_matrix
