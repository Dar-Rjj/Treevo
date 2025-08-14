import pandas as pd

def heuristics_v2(df):
    price_mad = (df['high'] - df['low']).rolling(window=20).apply(lambda x: x.mad(), raw=True)
    vol_ratio = price_mad / df['volume']
    ema_vol_ratio = vol_ratio.ewm(span=10, adjust=False).mean()
    heuristics_matrix = pd.Series(ema_vol_ratio, index=df.index)
    return heuristics_matrix
