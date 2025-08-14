def heuristics_v2(df):
    avg_price = (df['high'] + df['low'] + df['close']) / 3
    vol_avg_price_ratio = avg_price.rolling(window=10).std() / df['volume'].rolling(window=20).mean()
    heuristics_matrix = pd.Series(vol_avg_price_ratio, index=df.index)
    return heuristics_matrix
