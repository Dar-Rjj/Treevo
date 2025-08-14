def heuristics_v2(df):
    df['ema_close'] = df['close'].ewm(span=20, adjust=False).mean()
    df['daily_return'] = df['close'].pct_change()
    df['volatility'] = df['daily_return'].rolling(window=20).std()
    df['ema_volume'] = df['volume'].ewm(span=20, adjust=False).mean()
    df['time_weighted_price_change'] = df['price_change'].shift(-1).ewm(span=10, adjust=False).mean()
    heuristics_matrix = (df['time_weighted_price_change'] * df['ema_volume']) / df['volatility']
    return heuristics_matrix
