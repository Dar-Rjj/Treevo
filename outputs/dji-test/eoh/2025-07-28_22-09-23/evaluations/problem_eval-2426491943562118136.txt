def heuristics_v2(df):
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up / ema_down
    rsi = 100 - (100 / (1 + rs))
    wma_price_long = df['close'].rolling(window=50).apply(lambda x: (x * pd.Series(range(1, len(x)+1))).sum() / pd.Series(range(1, len(x)+1)).sum(), raw=True)
    price_to_wma_long = df['close'] / wma_price_long
    high_low_diff = df['high'] - df['low']
    heuristics_matrix = rsi * price_to_wma_long * high_low_diff
    return heuristics_matrix
