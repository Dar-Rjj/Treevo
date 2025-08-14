def heuristics_v2(df):
    def momentum(df, window=20):
        return df['close'].pct_change(window)

    def mean_reversion(df, window=20):
        return (df['close'] - df['close'].rolling(window=window).mean()) / df['close'].rolling(window=window).std()

    def volume_shock(df, window=5):
        return df['volume'].pct_change(window)

    df['momentum'] = momentum(df)
    df['mean_reversion'] = mean_reversion(df)
    df['volume_shock'] = volume_shock(df)

    heuristics_matrix = df[['momentum', 'mean_reversion', 'volume_shock']].copy()
    return heuristics_matrix
