def heuristics_v2(df):
    def sma_diff(price, fast=10, slow=50):
        sma_fast = price.rolling(window=fast).mean()
        sma_slow = price.rolling(window=slow).mean()
        return sma_fast - sma_slow

    def atr_volatility(high, low, close, window=14):
        tr = pd.Series(0, index=high.index)
        tr = tr.combine((high - low).abs(), max)
        tr = tr.combine((high - close.shift(1)).abs(), max)
        tr = tr.combine((low - close.shift(1)).abs(), max)
        atr = tr.rolling(window=window).mean()
        return (atr / close).rename('atr_volatility')

    sma_signal = sma_diff(df['close'])
    atr_vol = atr_volatility(df['high'], df['low'], df['close'])
    combined_factor = (sma_signal + atr_vol).rename('combined_factor')
    heuristics_matrix = combined_factor.rolling(window=30).mean().rename('heuristic_factor')

    return heuristics_matrix
