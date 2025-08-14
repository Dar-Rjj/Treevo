def heuristics_v2(df):
    df['ma_20_close'] = df['close'].rolling(window=20).mean()
    df['ma_50_close'] = df['close'].rolling(window=50).mean()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['std_log_returns_60'] = df['log_returns'].rolling(window=60).std()
    heuristics_matrix = (df['ma_20_close'] - df['ma_50_close']) / df['std_log_returns_60']
    return heuristics_matrix
