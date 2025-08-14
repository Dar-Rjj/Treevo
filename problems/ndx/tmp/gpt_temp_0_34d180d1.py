def heuristics_v2(df):
    close_change = df['close'].pct_change(periods=10)
    volatility = df['close'].rolling(window=50).std()
    volume_ma = df['volume'].rolling(window=252).mean()
    adjusted_factor = (close_change * volatility) - volume_ma
    heuristics_matrix = pd.Series(adjusted_factor, index=df.index)
    return heuristics_matrix
