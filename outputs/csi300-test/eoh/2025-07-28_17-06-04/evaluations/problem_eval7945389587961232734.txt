import pandas as pd

def heuristics_v2(df):
    daily_range = df['high'] - df['low']
    adj_daily_range = daily_range * df['volume'].apply(lambda x: max(1, x)).apply(lambda x: np.log(x))
    ewma = adj_daily_range.ewm(span=20, adjust=False).mean()
    heuristics_matrix = ewma
    return heuristics_matrix
