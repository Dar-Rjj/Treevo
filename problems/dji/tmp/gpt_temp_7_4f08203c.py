def heuristics_v2(df):
    # Calculate Intraday High-Low Spread
    df['high_low_spread'] = df['high'] - df['low']

    # Compute Intraday Range Weighted Average Price (IRWAP)
    df['irwap'] = (df['close'] * df['volume']).rolling(window=1, min_periods=1).sum() / df['volume'].rolling(window=1, min_periods=1).sum()

    # Evaluate IRWAP Difference
    df['irwap_diff'] = df['close'] - df['irwap']

    # Adjust Daily Return by Intraday Volatility
    df['daily_return'] = (df['close'] / df['close'].shift(1)) - 1
