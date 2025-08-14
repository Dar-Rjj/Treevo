def heuristics_v2(df):
    price_range = df['high'] - df['low']
    avg_volume = df['volume'].rolling(window=5).mean()
    range_vol_ratio = price_range / avg_volume
    ema_range_vol_ratio = range_vol_ratio.ewm(span=3, adjust=False).mean()
    heuristics_matrix = pd.Series(ema_range_vol_ratio, index=df.index)
    return heuristics_matrix
