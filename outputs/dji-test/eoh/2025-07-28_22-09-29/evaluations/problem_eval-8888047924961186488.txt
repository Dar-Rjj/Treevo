def heuristics_v2(df):
    def norm_volume(volume, periods=20):
        return (volume - volume.rolling(window=periods).min()) / (volume.rolling(window=periods).max() - volume.rolling(window=periods).min())

    def close_diff(price, periods=20):
        return price.rolling(window=periods).max() - price.rolling(window=periods).min()

    def wma(series, window=15):
        weights = np.arange(1, window + 1)
        wma_values = series.rolling(window=window).apply(lambda x: (x * weights).sum() / weights.sum(), raw=True)
        return wma_values

    norm_vol = norm_volume(df['volume'])
    close_difference = close_diff(df['close'])
    combined_factor = (norm_vol + close_difference).rename('combined_factor')
    heuristics_matrix = wma(combined_factor).rename('heuristic_factor')

    return heuristics_matrix
