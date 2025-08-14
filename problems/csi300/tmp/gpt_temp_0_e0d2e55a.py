def heuristics_v2(df):
    # Price Momentum: 50-day and 100-day simple moving average ratio
    sma_50 = df['close'].rolling(window=50).mean()
    sma_100 = df['close'].rolling(window=100).mean()
    sma_ratio = (sma_50 / sma_100) - 1
    
    # Volatility: Standard deviation of log returns over a 30-day period
    log_returns = np.log(df['close'] / df['close'].shift(1))
    volatility = log_returns.rolling(window=30).std()
    
    # Liquidity: Turnover ratio, calculated as the ratio of volume to the close price
    turnover_ratio = df['volume'] / df['close']
    
    # Volume-Weighted Average Price (VWAP)
    vwap = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
    
    # Composite heuristic
    heuristics_matrix = (sma_ratio + volatility + turnover_ratio + vwap) / 4
    return heuristics_matrix
