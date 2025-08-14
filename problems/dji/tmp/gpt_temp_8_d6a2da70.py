def heuristics_v2(df):
    def momentum(price, periods=14):
        return price / price.shift(periods) - 1

    def atr(high, low, close, periods=14):
        tr = pd.DataFrame(index=high.index)
        tr['hl'] = abs(high - low)
        tr['hc'] = abs(high - close.shift(1))
        tr['lc'] = abs(low - close.shift(1))
        tr['tr'] = tr.max(axis=1)
        return tr['tr'].ewm(span=periods, adjust=False).mean()

    mom_signal = momentum(df['close'])
    atr_signal = atr(df['high'], df['low'], df['close'])
    combined_factor = (mom_signal + atr_signal).rename('combined_factor')
    heuristics_matrix = combined_factor.ewm(span=20, adjust=False).mean().rename('heuristic_factor')

    return heuristics_matrix
