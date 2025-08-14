def heuristics_v2(df):
    df['price_change'] = df['close'].pct_change()
    df['log_return'] = np.log(df['close']) - np.log(df['close'].shift(1))
    df['volatility'] = df['log_return'].rolling(window=20).std()
    df['vma'] = (df['volume'] * df['close']).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
    heuristics_matrix = (df['price_change'].shift(-1) * df['vma']) / df['volatility']
    return heuristics_matrix
