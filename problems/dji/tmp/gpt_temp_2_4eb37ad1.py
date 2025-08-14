def heuristics_v2(df):
    ema_price = (df['high'] - df['low']).ewm(span=15, adjust=False).mean()
    vol_vol_ratio = ema_price / df['volume'].rolling(window=25).std()
    ema_vol_vol_ratio = vol_vol_ratio.ewm(span=35, adjust=False).mean()
    heuristics_matrix = pd.Series(ema_vol_vol_ratio, index=df.index)
    return heuristics_matrix
