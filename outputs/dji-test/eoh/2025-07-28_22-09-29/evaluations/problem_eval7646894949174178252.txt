def heuristics_v2(df):
    def rsi(series, periods=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=periods).mean()
        avg_loss = loss.rolling(window=periods).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def obv(close, volume):
        obv = (np.sign(close.diff()) * volume).cumsum()
        return obv

    rsi_signal = rsi(df['close'])
    obv_signal = obv(df['close'], df['volume'])
    combined_factor = (rsi_signal + obv_signal).rename('combined_factor')
    heuristics_matrix = combined_factor.ewm(span=20, adjust=False).mean().rename('heuristic_factor')

    return heuristics_matrix
