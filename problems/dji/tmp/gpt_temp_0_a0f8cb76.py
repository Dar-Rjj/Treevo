def heuristics_v2(df):
    ema_price = (df['high'] + df['low']) / 2
    ema_price = ema_price.ewm(span=15, adjust=False).mean()
    vol_vol_ratio = ema_price / df['volume'].rolling(window=30).std()
    ema_vol_vol_ratio = vol_vol_ratio.ewm(span=40, adjust=False).mean()
    heuristics_matrix = pd.Series(ema_vol_vol_ratio, index=df.index)
    return heuristics_matrix
