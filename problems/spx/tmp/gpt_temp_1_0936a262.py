def heuristics_v2(df):
    # Calculate Short-term (5 days) and Long-term (20 days) moving averages of the close prices
    df['short_ma'] = df['close'].rolling(window=5).mean()
    df['long_ma'] = df['close'].rolling(window=20).mean()
    df['ma_crossover'] = df['short_ma'] - df['long_ma']
    
    # Calculate EMA using short-term (5 days) and long-term (20 days) periods on close prices
    df['short_ema'] = df['close'].ewm(span=5, adjust=False).mean()
    df['long_ema'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_crossover'] = df['short_ema'] - df['long_ema']
    
    # Compute the return from t-n to t, where n is 1, 5, 10, or 20 days
