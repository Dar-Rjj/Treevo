def heuristics_v2(df):
    df['price_change'] = df['close'].pct_change()
    avg_price_change = df['price_change'].rolling(window=10).mean()
    heuristics_matrix = avg_price_change * np.log(df['volume'])
    return heuristics_matrix
