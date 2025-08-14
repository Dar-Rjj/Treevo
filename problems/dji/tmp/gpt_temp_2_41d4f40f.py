def heuristics_v2(df):
    def vwap(price, volume, window=10):
        return (price * volume).rolling(window=window).sum() / volume.rolling(window=window).sum()

    def daily_returns(series):
        return series.pct_change()

    def weighted_moving_average(series, weights, window=20):
        return (series * weights).rolling(window=window).sum() / weights.rolling(window=window).sum()

    vwap_signal = vwap(df['close'], df['volume'])
    daily_ret = daily_returns(df['close'])
    std_dev = daily_ret.rolling(window=20).std()
    combined_factor = (vwap_signal + std_dev).rename('combined_factor')
    weights = np.linspace(1, 0, 20)
    heuristics_matrix = weighted_moving_average(combined_factor, weights).rename('heuristic_factor')

    return heuristics_matrix
