def heuristics_v2(df):
    rsi_window = 14
    ema_volume_window = 14
    long_term_volatility_window = 10

    # Calculate RSI
    delta = df['close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    # Calculate EMA of volume
    df['ema_volume'] = df['volume'].ewm(span=ema_volume_window, adjust=False).mean()

    # Calculate the logarithmic difference between the highest high and lowest low
    max_high = df['high'].rolling(window=long_term_volatility_window).max()
    min_low = df['low'].rolling(window=long_term_volatility_window).min()
    log_diff_high_low = np.log(max_high / min_low)

    heuristics_matrix = rsi * (df['volume'] / df['ema_volume']) * log_diff_high_low
    return heuristics_matrix
