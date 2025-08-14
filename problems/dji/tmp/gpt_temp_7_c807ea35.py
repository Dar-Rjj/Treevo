def heuristics_v2(df):
    df['log_return'] = np.log(df['close']).diff(periods=10)
    df['cumulative_volume'] = df['volume'].rolling(window=10).sum()
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=10).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=10).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    heuristics_matrix = df['log_return'] * df['cumulative_volume'] * (df['rsi'] / 100)
    return heuristics_matrix
