def heuristics_v2(df):
    # Price-to-volume ratio
    price_to_volume = df['close'] / df['volume']
    
    # Momentum: 10-day simple moving average of daily returns
    returns = df['close'].pct_change()
    sma_returns = returns.rolling(window=10).mean()
    
    # Volatility: Standard deviation of log-returns over 20 days
    log_returns = np.log(df['close'] / df['close'].shift(1))
    volatility = log_returns.rolling(window=20).std()
    
    # Combine factors into a heuristics matrix using a weighted sum
    heuristics_matrix = (price_to_volume + sma_returns - volatility) / 3
    
    return heuristics_matrix
