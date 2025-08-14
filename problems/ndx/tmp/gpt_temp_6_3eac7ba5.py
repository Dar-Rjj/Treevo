def heuristics_v2(df):
    # Calculate the 20-day and 100-day simple moving averages of the close price
    sma_20 = df['close'].rolling(window=20).mean()
    sma_100 = df['close'].rolling(window=100).mean()

    # Calculate the ratio between the SMAs
    sma_ratio = sma_20 / sma_100

    # Calculate the daily return
    df['Return'] = df['close'].pct_change()

    # Calculate the 30-day standard deviation of daily returns
    std_30 = df['Return'].rolling(window=30).std()

    # Calculate the 30-day Average True Range (ATR)
    tr = pd.DataFrame({
        'H-L': df['high'] - df['low'],
        'H-Cp': abs(df['high'] - df['close'].shift(1)),
        'L-Cp': abs(df['low'] - df['close'].shift(1))
    }).max(axis=1)
    atr_30 = tr.rolling(window=30).mean()

    # Combine the 30-day standard deviation of daily returns and the 30-day ATR
    combined_metric = (std_30 + atr_30) / 2

    # Generate the heuristic matrix by multiplying the SMA ratio with the combined metric
    heuristics_matrix = sma_ratio * combined_metric

    return heuristics_matrix
