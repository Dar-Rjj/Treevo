def heuristics_v2(df):
    price_avg = (df['high'] + df['low']) / 2
    vol_std = df['volume'].rolling(window=10).std()
    pv_vol_ratio = price_avg / vol_std
    ema_pv_vol_ratio = pv_vol_ratio.ewm(span=20, adjust=False).mean()
    heuristics_matrix = pd.Series(ema_pv_vol_ratio, index=df.index)
    return heuristics_matrix
