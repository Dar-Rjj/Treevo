import uniform_filter as wma

    def wma_diff(price, fast=8, slow=20):
        wma_fast = wma(price, size=fast)
        wma_slow = wma(price, size=slow)
        return wma_fast - wma_slow

    def roc(series, periods=14):
        return series.pct_change(periods)

    wma_signal = wma_diff(df['close'])
    roc_low = roc(df['low'])
    combined_factor = (wma_signal + roc_low).rename('combined_factor')
    heuristics_matrix = combined_factor.rolling(window=15, min_periods=1).median().rename('heuristic_factor')

    return heuristics_matrix
