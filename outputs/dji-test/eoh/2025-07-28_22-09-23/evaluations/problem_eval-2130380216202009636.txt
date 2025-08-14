def heuristics_v2(df):
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=6, adjust=False).mean()
    ema_down = down.ewm(com=6, adjust=False).mean()
    rs = ema_up / ema_down
    rsi = 100 - (100 / (1 + rs))
    avg_price_long = df['close'].rolling(window=20).mean()
    price_to_avg_long = df['close'] / avg_price_long
    high_low_diff_log = (df['high'] - df['low']).apply(lambda x: np.log1p(x))
    heuristics_matrix = rsi * price_to_avg_long * high_low_diff_log
    return heuristics_matrix
