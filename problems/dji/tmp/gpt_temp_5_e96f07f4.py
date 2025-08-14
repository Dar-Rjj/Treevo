def heuristics_v2(df):
    df['price_change'] = np.log(df['close']).diff()
    df['volume_ma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    df['atr'] = (df['high'] - df['low']).rolling(window=21).mean()
    heuristics_matrix = (df['price_change'].shift(-1) * df['volume_ratio']) / df['atr']
    return heuristics_matrix
