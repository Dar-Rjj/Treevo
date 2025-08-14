def heuristics_v2(df):
    # Calculate On-Balance Volume (OBV)
    obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()

    # Calculate Chaikin Money Flow (CMF)
    money_flow_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    money_flow_volume = money_flow_multiplier * df['volume']
    cmf = money_flow_volume.rolling(window=20).sum() / df['volume'].rolling(window=20).sum()

    # Combine OBV and CMF into a single heuristics measure
    heuristics_matrix = (obv + cmf) / 2

    return heuristics_matrix
