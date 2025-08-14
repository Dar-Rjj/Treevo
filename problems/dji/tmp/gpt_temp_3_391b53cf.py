def heuristics_v2(df):
    # Calculate simple moving averages (SMA)
    df['SMA_5'] = df['close'].rolling(window=5).mean()
    df['SMA_10'] = df['close'].rolling(window=10).mean()
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()

    # Calculate the difference between current close and SMA
    df['SMA_5_diff'] = df['close'] - df['SMA_5']
    df['SMA_10_diff'] = df['close'] - df['SMA_10']
    df['SMA_20_diff'] = df['close'] - df['SMA_20']
    df['SMA_50_diff'] = df['close'] - df['SMA_50']

    # Calculate returns over various look-back periods
