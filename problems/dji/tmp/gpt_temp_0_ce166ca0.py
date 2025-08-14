def heuristics_v2(df):
    avg_price = (df['high'] + df['low']) / 2
    avg_price_7d = avg_price.rolling(window=7).mean()
    vol_std = df['volume'].rolling(window=15).std()
    pv_mom_ratio = avg_price_7d / vol_std
    sma_pv_mom_ratio = pv_mom_ratio.rolling(window=25).mean()
    heuristics_matrix = pd.Series(sma_pv_mom_ratio, index=df.index)
    return heuristics_matrix
