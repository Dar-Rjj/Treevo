import pandas as pd

def heuristics_v2(df):
    daily_range = df['high'] - df['low']
    pct_change_close = df['close'].pct_change().fillna(0)
    abs_pct_change_close = abs(pct_change_close)
    weighted_factor = df['volume'] * (1 + abs_pct_change_close)
    ewm_weighted_average = (daily_range * weighted_factor).ewm(span=20).mean() / weighted_factor.ewm(span=20).mean()
    return heuristics_matrix
