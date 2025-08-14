def heuristics_v2(df):
    avg_price = (df['high'] + df['low']) / 2
    vol_std = df['volume'].rolling(window=5).std()
    pv_vol_ratio = avg_price / vol_std
    ema_pv_vol_ratio = pv_vol_ratio.ewm(span=20, adjust=False).mean()
    heuristics_matrix = pd.Series(ema_pv_vol_ratio, index=df.index)
    return heuristics_matrix
