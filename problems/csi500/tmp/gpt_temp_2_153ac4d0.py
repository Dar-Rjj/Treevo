def heuristics_v2(df):
    # Calculate SMA of closing prices for 5, 10, and 20 days
    df['SMA_5'] = df['close'].rolling(window=5).mean()
    df['SMA_10'] = df['close'].rolling(window=10).mean()
    df['SMA_20'] = df['close'].rolling(window=20).mean()

    # Compute the difference between the current close and each SMA
    df['SMA_diff_5'] = df['close'] - df['SMA_5']
    df['SMA_diff_10'] = df['close'] - df['SMA_10']
    df['SMA_diff_20'] = df['close'] - df['SMA_20']

    # Calculate the percentage change in closing prices over the last 5 and 10 days
    df['pct_change_5'] = df['close'].pct_change(periods=5)
    df['pct_change_10'] = df['close'].pct_change(periods=10)

    # Calculate the RSI (Relative Strength Index) over a 14-day period
    df['RSI_14'] = RSI(df['close'], timeperiod=14)

    # Calculate the ratio of current volume to the 20-day average volume
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()

    # Determine if the volume is above or below its 20-day moving average
    df['volume_above_avg'] = (df['volume'] > df['volume'].rolling(window=20).mean()).astype(int)

    # Calculate the standard deviation of daily returns over the last 20 days
