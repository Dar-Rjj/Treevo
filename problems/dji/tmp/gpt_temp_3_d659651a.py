def heuristics_v2(df):
    volume_price_ratio = df['volume'] / df['close']
    delta = volume_price_ratio.diff(1)
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    roll_up = up.rolling(window=14).mean()
    roll_down = down.abs().rolling(window=14).mean()
    RS = roll_up / roll_down
    RSI = 100.0 - (100.0 / (1.0 + RS))
    return heuristics_matrix
