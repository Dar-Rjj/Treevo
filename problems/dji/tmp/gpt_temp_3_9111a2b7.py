def heuristics_v2(df):
    # Momentum Indicators
    N = 14
    df['momentum_close'] = df['close'] - df['close'].rolling(window=N).mean()
    df['pct_change_close'] = df['close'].pct_change()

    # Volatility Indicators
    M = 20
    P = 14
    df['daily_returns'] = df['close'].pct_change()
