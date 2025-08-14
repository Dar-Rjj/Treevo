def heuristics_v2(df):
    df['smoothed_price_change'] = df['close'].pct_change().ewm(span=14, adjust=False).mean()
    df['volume_rsi'] = 100 - (100 / (1 + (df['volume'].rolling(window=14).mean() / df['volume'].rolling(window=14).std())))
    df['modified_atr'] = df[['high', 'low']].rolling(window=28).max() - df[['high', 'low']].rolling(window=28).min()
    heuristics_matrix = (df['smoothed_price_change'].shift(-1) * df['volume_rsi']) / df['modified_atr']['high']
    return heuristics_matrix
