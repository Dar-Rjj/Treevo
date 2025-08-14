def heuristics_v2(df):
    def wma_diff(price, fast=10, slow=30):
        weights_fast = pd.Series(range(1, fast + 1))
        wma_fast = (price.rolling(window=fast).apply(lambda x: (x * weights_fast).sum() / weights_fast.sum(), raw=True))
        weights_slow = pd.Series(range(1, slow + 1))
        wma_slow = (price.rolling(window=slow).apply(lambda x: (x * weights_slow).sum() / weights_slow.sum(), raw=True))
        return wma_fast - wma_slow

    def std_volume(series, periods=20):
        return series.rolling(window=periods).std()

    wma_signal = wma_diff(df['close'])
    std_vol = std_volume(df['volume'])
    combined_factor = (wma_signal + std_vol).rename('combined_factor')
    heuristics_matrix = combined_factor.ewm(span=20, adjust=False).mean().rename('heuristic_factor')

    return heuristics_matrix
