def heuristics_v2(df):
    avg_price = (df['high'] + df['low'] + df['close']) / 3
    volume_avg_ratio = df['volume'] / avg_price
    log_volume_avg_ratio = np.log(volume_avg_ratio)
    heuristics_matrix = log_volume_avg_ratio.ewm(span=20, adjust=False).mean()
    return heuristics_matrix
