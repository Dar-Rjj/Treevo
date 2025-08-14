def heuristics_v2(df):
    # Define liquidity measures
    df['avg_vol_30'] = df['volume'].rolling(window=30).mean()
    df['trading_range'] = df['high'] - df['low']
    
    # Define dynamic volatility measures
    df['historical_volatility_30'] = df['close'].pct_change().rolling(window=30).std()
    df['intraday_volatility'] = (df['high'] - df['low']) / ((df['high'] + df['low']) / 2)
    
    # Combine liquidity and volatility
    liquidity_weight = 0.5  # Adjust based on recent market conditions
    volatility_weight = 0.5  # Adjust based on recent market conditions
    df['liquidity_volatility'] = (liquidity_weight * df['avg_vol_30'] + 
                                 volatility_weight * df['historical_volatility_30'])
    
    # Volume adjustments
    df['volume_weighted_return'] = (df['close'].pct_change() * df['volume']) / df['volume'].rolling(window=30).mean()
