def heuristics_v2(df):
    # Rate of Change factor: 21-day percentage change in close price
    roc = df['close'].pct_change(21)
    
    # Volatility Measure: Standard Deviation of log returns over a 21-day window
    log_returns = np.log(df['close'] / df['close'].shift(1))
    volatility = log_returns.rolling(window=21).std()
    
    # Liquidity Measure: Traded amount divided by the moving average of volume
    ma_volume = df['volume'].rolling(window=50).mean()
    liquidity = df['amount'] / ma_volume
    
    # Composite heuristic
    heuristics_matrix = (roc + volatility + liquidity) / 3
    return heuristics_matrix
