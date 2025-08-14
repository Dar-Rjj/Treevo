import pandas as pd

def heuristics_v2(df):
    ema_close = df['close'].ewm(span=20, adjust=False).mean()
    ema_volume = df['volume'].ewm(span=10, adjust=False).mean()
    vol_ratio = df['volume'] / ema_volume
    ln_hl_ratio = (df['high'] / df['low']).apply(lambda x: math.log(x))
    heuristics_matrix = 0.4 * ema_close + 0.3 * vol_ratio + 0.3 * ln_hl_ratio
    return heuristics_matrix
