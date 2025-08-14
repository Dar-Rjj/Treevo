import pandas as pd

def heuristics_v2(df):
    ma_short = df['close'].rolling(window=10).mean()
    ma_long = df['close'].rolling(window=50).mean()
    delta = df['close'].diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    roll_up = up.rolling(window=14).mean()
    roll_down = down.abs().rolling(window=14).mean()
    rs = roll_up / roll_down
    rsi = 100.0 - (100.0 / (1.0 + rs))
    price_to_volume_log = (df['close'] / df['volume']).apply(lambda x: 0 if x <= 0 else np.log(x))
    heuristics_matrix = (ma_short - ma_long) * (100 - rsi) + price_to_volume_log
    return heuristics_matrix
