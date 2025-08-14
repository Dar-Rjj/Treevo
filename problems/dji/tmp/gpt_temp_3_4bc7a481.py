def heuristics_v2(df):
    # Calculate Simple Moving Averages
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    df['SMA_100'] = df['close'].rolling(window=100).mean()
    df['SMA_200'] = df['close'].rolling(window=200).mean()

    # Subtract Longer SMA from Shorter SMA
    df['SMA_100_200_diff'] = df['SMA_100'] - df['SMA_200']
    df['SMA_50_100_diff'] = df['SMA_50'] - df['SMA_100']

    # Calculate Intraday Price Range Percentage
    df['intraday_range_pct'] = (df['high'] - df['low']) / df['low'] * 100

    # Calculate Intraday Momentum
    df['intraday_high_return'] = (df['high'] - df['open']) / df['open']
