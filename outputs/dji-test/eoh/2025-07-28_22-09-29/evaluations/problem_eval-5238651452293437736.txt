def heuristics_v2(df):
    def ppo(price, fast=12, slow=26, signal=9):
        ema_fast = price.ewm(span=fast, adjust=False).mean()
        ema_slow = price.ewm(span=slow, adjust=False).mean()
        ppo = ((ema_fast - ema_slow) / ema_slow) * 100
        return ppo

    def mfi(high, low, close, volume, n=14):
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        positive_money_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_money_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        money_ratio = positive_money_flow.rolling(window=n).sum() / negative_money_flow.rolling(window=n).sum()
        mfi = 100 - (100 / (1 + money_ratio))
        return mfi

    def wma(series, window=20):
        weights = pd.Series(range(1, window + 1))
        wma = series.rolling(window=window).apply(lambda prices: (prices * weights).sum() / weights.sum(), raw=True)
        return wma

    ppo_signal = ppo(df['close'])
    mfi_signal = mfi(df['high'], df['low'], df['close'], df['volume'])
    combined_factor = (ppo_signal + mfi_signal).rename('combined_factor')
    heuristics_matrix = wma(combined_factor, window=20).rename('heuristic_factor')

    return heuristics_matrix
