def heuristics_v2(df):
    # Calculate Volume-Weighted Short-Term Moving Average
    short_term = 10
    df['vwma_short'] = (df['close'] * df['volume']).rolling(window=short_term).sum() / df['volume'].rolling(window=short_term).sum()

    # Calculate Volume-Weighted Long-Term Moving Average
    long_term = 50
    df['vwma_long'] = (df['close'] * df['volume']).rolling(window=long_term).sum() / df['volume'].rolling(window=long_term).sum()
    
    # Calculate Raw Daily Return
    df['daily_return'] = df['close'].pct_change()
