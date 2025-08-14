def heuristics_v2(df):
    # Calculate Moving Averages
    df['50_day_EMA'] = df['close'].ewm(span=50, adjust=False).mean()
    df['100_day_EMA'] = df['close'].ewm(span=100, adjust=False).mean()
    df['200_day_EMA'] = df['close'].ewm(span=200, adjust=False).mean()

    df['50_day_SMA'] = df['close'].rolling(window=50).mean()
    df['100_day_SMA'] = df['close'].rolling(window=100).mean()
    df['200_day_SMA'] = df['close'].rolling(window=200).mean()

    # Subtract Longer MA from Shorter MA
    df['EMA_100_200'] = df['100_day_EMA'] - df['200_day_EMA']
    df['EMA_50_100'] = df['50_day_EMA'] - df['100_day_EMA']
    df['SMA_100_200'] = df['100_day_SMA'] - df['200_day_SMA']
    df['SMA_50_100'] = df['50_day_SMA'] - df['100_day_SMA']

    # Calculate Intraday Return
    df['intraday_return'] = (df['close'] - df['open']) / df['open']
