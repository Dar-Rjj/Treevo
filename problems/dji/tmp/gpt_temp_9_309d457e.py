def heuristics_v2(df):
    df['log_return'] = np.log(df['close']) - np.log(df['close'].shift(1))
    df['volume_mean'] = df['volume'].rolling(window=10).mean()
    df['volume_std'] = df['volume'].rolling(window=10).std()
    df['volume_rsi'] = 100 - (100 / (1 + (df['volume_mean'] / df['volume_std'])))
    df['atr'] = df[['high', 'low']].diff(axis=1).iloc[:,0].rolling(window=10).mean()
    heuristics_matrix = (df['log_return'].shift(-1) * df['volume_rsi']) / df['atr']
    return heuristics_matrix
