def heuristics_v2(df):
    adj_avg_price = (df['high'] + df['low'] + 2 * df['close']) / 4
    volume_adj_avg_ratio = df['volume'] / adj_avg_price
    log_return = df['close'].pct_change().apply(lambda x: np.log(1 + x))
    combined_factor = volume_adj_avg_ratio * log_return
    heuristics_matrix = combined_factor.ewm(span=30, adjust=False).mean()
    return heuristics_matrix
