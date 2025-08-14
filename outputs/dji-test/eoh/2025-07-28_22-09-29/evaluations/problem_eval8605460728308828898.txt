def heuristics_v2(df):
    def ema_difference(price, fast=10, slow=30):
        ema_fast = price.ewm(span=fast, adjust=False).mean()
        ema_slow = price.ewm(span=slow, adjust=False).mean()
        return ema_fast - ema_slow

    def atr(high, low, close, period=14):
        tr = pd.DataFrame(index=high.index)
        tr['tr1'] = high - low
        tr['tr2'] = (high - close.shift(1)).abs()
        tr['tr3'] = (low - close.shift(1)).abs()
        tr['true_range'] = tr[['tr1', 'tr2', 'tr3']].max(axis=1)
        atr_val = tr['true_range'].ewm(span=period, adjust=False).mean()
        return atr_val

    ema_momentum = ema_difference(df['close'])
    avg_true_range = atr(df['high'], df['low'], df['close'])
    combined_factor = (ema_momentum + avg_true_range).rename('combined_factor')
    heuristics_matrix = combined_factor.ewm(span=18, adjust=False).mean().rename('heuristic_factor')

    return heuristics_matrix
