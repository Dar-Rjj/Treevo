def heuristics_v2(df):
    # Simple Moving Average (SMA)
    df['SMA_10'] = df['close'].rolling(window=10).mean()
    df['SMA_20'] = df['close'].rolling(window=20).mean()

    # Exponential Moving Average (EMA)
    df['EMA_9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['EMA_21'] = df['close'].ewm(span=21, adjust=False).mean()

    # Rate of Change (ROC)
    df['ROC_12'] = df['close'].pct_change(periods=12)
    df['ROC_25'] = df['close'].pct_change(periods=25)

    # Standard Deviation of daily returns
