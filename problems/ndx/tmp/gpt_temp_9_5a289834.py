def heuristics_v2(df):
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    momentum = typical_price.pct_change(periods=20)
    atr = df['high'].rolling(window=14).max() - df['low'].rolling(window=14).min()
    combined_factor = momentum * atr
    smoothed_factor = combined_factor.ewm(span=5).mean().dropna()
    return heuristics_matrix
