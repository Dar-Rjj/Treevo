def heuristics_v2(df):
    money_flow_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    money_flow_volume = money_flow_multiplier * df['volume']
    adl = money_flow_volume.cumsum()
    adl_ema = adl.ewm(span=20, adjust=False).mean()
    tr1 = abs(df['high'] - df['low'])
    tr2 = abs(df['high'] - df['close'].shift())
    tr3 = abs(df['low'] - df['close'].shift())
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=30).mean()
    heuristics_matrix = (adl_ema / atr).fillna(0)
    return heuristics_matrix
