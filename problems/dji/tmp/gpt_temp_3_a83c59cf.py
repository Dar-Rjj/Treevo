import pandas as pd

def heuristics_v2(df):
    median_price = df[['high', 'low']].median(axis=1)
    mad_vol = df['volume'].rolling(window=10).apply(lambda x: (x - x.mean()).abs().mean(), raw=True)
    vol_vol_ratio = median_price / mad_vol
    ewma_vol_vol_ratio = vol_vol_ratio.ewm(span=30).mean()
    heuristics_matrix = pd.Series(ewma_vol_vol_ratio, index=df.index)
    return heuristics_matrix
