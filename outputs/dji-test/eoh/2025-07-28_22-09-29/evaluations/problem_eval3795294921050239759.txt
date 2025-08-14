def heuristics_v2(df):
    def ema_ratio(price, short=12, long=26):
        ema_short = price.ewm(span=short, adjust=False).mean()
        ema_long = price.ewm(span=long, adjust=False).mean()
        return ema_short / ema_long

    def atr(price_high, price_low, close, window=14):
        tr = pd.concat([(price_high - price_low), 
                        (close.shift() - price_high).abs(), 
                        (close.shift() - price_low).abs()], axis=1).max(axis=1)
        atr_value = tr.rolling(window=window).mean()
        return atr_value / close

    ema_signal = ema_ratio(df['close'])
    atr_signal = atr(df['high'], df['low'], df['close'])
    combined_factor = (ema_signal + atr_signal).rename('combined_factor')
    weights = np.arange(1, 31)  # Weights for the WMA
    heuristics_matrix = combined_factor.rolling(window=30, min_periods=1).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True).rename('heuristic_factor')

    return heuristics_matrix
