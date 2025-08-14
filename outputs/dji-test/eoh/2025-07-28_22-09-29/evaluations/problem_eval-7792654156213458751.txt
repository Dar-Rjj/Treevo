def heuristics_v2(df):
    def momentum(price, periods=10):
        return price.pct_change(periods)

    def roc(series, periods=14):
        return series.pct_change(periods)

    momentum_low = momentum(df['low'])
    roc_volume = roc(df['volume'])
    combined_factor = (momentum_low + roc_volume).rename('combined_factor')
    weights = np.arange(1, 21)
    heuristics_matrix = combined_factor.rolling(window=20).apply(lambda x: np.average(x, weights=weights), raw=True).rename('heuristic_factor')

    return heuristics_matrix
