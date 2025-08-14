def heuristics_v2(df):
    momentum = df['close'].rolling(window=20).mean() - df['close'].rolling(window=50).mean()
    liquidity = df['volume'].rolling(window=30).mean() / df['shares_outstanding']
    heuristics_matrix = (momentum + liquidity)
    return heuristics_matrix
