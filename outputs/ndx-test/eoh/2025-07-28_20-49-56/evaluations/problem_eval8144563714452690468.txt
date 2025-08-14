def heuristics_v2(df):
    ema_short = df['close'].ewm(span=10, adjust=False).mean()
    ema_long = df['close'].ewm(span=30, adjust=False).mean()
    ema_signal = ema_short - ema_long
    trading_range = df['high'] - df['low']
    volume_change_log = df['volume'].apply(lambda x: np.log(x) if x > 0 else 0).diff().fillna(0)
    heuristics_matrix = ema_signal + trading_range + volume_change_log
    return heuristics_matrix
