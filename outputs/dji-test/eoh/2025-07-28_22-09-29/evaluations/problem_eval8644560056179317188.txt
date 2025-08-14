def heuristics_v2(df):
    def sma_diff(price, short=10, long=50):
        sma_short = price.rolling(window=short).mean()
        sma_long = price.rolling(window=long).mean()
        return sma_short - sma_long

    def atr(data, periods=14):
        high_low = data['high'] - data['low']
        high_close = (data['high'] - data['close'].shift()).abs()
        low_close = (data['low'] - data['close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return ranges.rolling(window=periods).mean()

    sma_signal = sma_diff(df['close'])
    atr_signal = atr(df)
    combined_factor = (sma_signal + atr_signal).rename('combined_factor')
    heuristics_matrix = combined_factor.ewm(span=20, adjust=False).mean().rename('heuristic_factor')

    return heuristics_matrix
