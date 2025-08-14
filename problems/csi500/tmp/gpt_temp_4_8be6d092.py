def heuristics_v2(df):
    # Calculate Simple Moving Averages (SMA)
    df['SMA_5'] = df['close'].rolling(window=5).mean()
    df['SMA_10'] = df['close'].rolling(window=10).mean()
    df['SMA_20'] = df['close'].rolling(window=20).mean()

    # Compute the difference between the current close and each SMA
    df['SMA_5_diff'] = df['close'] - df['SMA_5']
    df['SMA_10_diff'] = df['close'] - df['SMA_10']
    df['SMA_20_diff'] = df['close'] - df['SMA_20']

    # Calculate the percentage change in closing prices over the last 5 and 10 days
    df['pct_change_5'] = df['close'].pct_change(periods=5) * 100
    df['pct_change_10'] = df['close'].pct_change(periods=10) * 100

    # Calculate the RSI (Relative Strength Index) over a 14-day period
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # Calculate the ratio of current volume to the 20-day average volume
    df['avg_volume_20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio_20'] = df['volume'] / df['avg_volume_20']

    # Determine if the volume is above or below its 20-day moving average
    df['volume_trend'] = (df['volume'] > df['avg_volume_20']).astype(int)

    # Compute Volume Moving Average: Use Volume with a 10-day window
    df['volume_MA_10'] = df['volume'].rolling(window=10).mean()

    # Measure Volume Momentum: Change in Aggregated Volume (Sum of past n days' volume)
    df['volume_momentum'] = df['volume'].rolling(window=5).sum() / df['volume'].rolling(window=10).sum()

    # Calculate the standard deviation of daily returns over the last 20 days
