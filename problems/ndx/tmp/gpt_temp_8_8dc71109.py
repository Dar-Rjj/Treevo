def heuristics_v2(df):
    n = 14
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=n).mean()
    avg_loss = loss.rolling(window=n).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    log_rsi = np.log(rsi)
    avg_volume = df['volume'].rolling(window=n).mean()
    vol_ratio = df['volume'] / avg_volume
    heuristics_matrix = log_rsi * vol_ratio
    return heuristics_matrix
