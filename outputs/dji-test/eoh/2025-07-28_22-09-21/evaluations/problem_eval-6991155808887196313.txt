def heuristics_v2(df):
    df['log_returns'] = np.log(df['close']) - np.log(df['close'].shift(1))
    df['ema'] = df['close'].ewm(span=14, adjust=False).mean()
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['std_log_returns'] = df['log_returns'].rolling(window=30).std()
    heuristics_matrix = (df['ema'] * df['rsi']) / df['std_log_returns']
    return heuristics_matrix
