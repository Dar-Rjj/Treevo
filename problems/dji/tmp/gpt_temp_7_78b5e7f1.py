def heuristics_v2(df):
    def pct_change(price, periods=5):
        return price.pct_change(periods=periods)

    def roc(series, periods=14):
        return (series - series.shift(periods)) / series.shift(periods) * 100

    pct_price = pct_change(df['close'], periods=5)
    roc_volume = roc(df['volume'], periods=14)
    combined_factor = (pct_price + roc_volume).rename('combined_factor')
    
    weights = np.arange(1, 21)
    weights = weights / weights.sum()
    heuristics_matrix = combined_factor.rolling(window=20).apply(lambda x: np.dot(x, weights), raw=True).rename('heuristic_factor')

    return heuristics_matrix
