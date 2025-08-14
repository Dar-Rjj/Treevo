def heuristics_v2(df):
    # Calculate simple moving averages for close price
    sma_5 = df['close'].rolling(window=5).mean()
    sma_10 = df['close'].rolling(window=10).mean()
    sma_20 = df['close'].rolling(window=20).mean()

    # Momentum factor: current close price minus 5-day SMA
    momentum_5 = df['close'] - sma_5

    # Volatility factor: standard deviation of daily returns over the last 10 days
    daily_returns = df['close'].pct_change()
    volatility_10 = daily_returns.rolling(window=10).std()

    # Volume change factor: percentage change in volume from the previous day
    volume_change = df['volume'].pct_change()

    # Combine all heuristics into a DataFrame
    heuristics_matrix = pd.DataFrame({
        'momentum_5': momentum_5,
        'volatility_10': volatility_10,
        'volume_change': volume_change
    })

    return heuristics_matrix
