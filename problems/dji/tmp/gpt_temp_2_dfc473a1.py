def heuristics_v2(df):
    df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['High10'] = df['high'].rolling(window=10).max()
    df['avg_volume_30'] = df['volume'].rolling(window=30).mean()
    heuristics_matrix = (df['High10'] - df['EMA20']) * np.log(df['volume'] / df['avg_volume_30'])
    return heuristics_matrix
