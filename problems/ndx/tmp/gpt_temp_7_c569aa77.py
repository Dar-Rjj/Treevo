def heuristics_v2(df):
    vwap_last_20 = ((df['close'] + df['open']) / 2 * df['volume']).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
    vwap_next_20 = ((df['close'].shift(-20) + df['open'].shift(-20)) / 2 * df['volume'].shift(-20)).rolling(window=20).sum() / df['volume'].shift(-20).rolling(window=20).sum()
    factor_values = vwap_last_20 / vwap_next_20
    smoothed_factor = factor_values.ewm(span=20, adjust=False).mean()
    return heuristics_matrix
