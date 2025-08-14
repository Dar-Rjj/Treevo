def heuristics_v2(df):
    atr = df[['high', 'low', 'close']].apply(lambda x: np.max(x) - np.min(x), axis=1)
    atr_14 = atr.rolling(window=14).mean()
    atr_roc = atr_14.pct_change(periods=5)
    ema_20_close = df['close'].ewm(span=20, adjust=False).mean()
    heuristics_matrix = atr_roc * ema_20_close
    return heuristics_matrix
