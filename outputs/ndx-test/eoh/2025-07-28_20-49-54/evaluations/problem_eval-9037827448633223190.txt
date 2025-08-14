def heuristics_v2(df):
    # Momentum factor: 5-day return
    df['momentum'] = (df['close'] / df['close'].shift(5)) - 1
    
    # Volatility factor: standard deviation of daily returns over the last 30 days
    df['volatility'] = df['close'].pct_change().rolling(window=30).std()
    
    # Volume factor: average volume over the last 10 days divided by the previous day's volume
    df['volume_factor'] = df['volume'].rolling(window=10).mean() / df['volume'].shift(1)
    
    # Combine factors into a single heuristic score
    df['heuristic_score'] = (df['momentum'] * 0.4) - (df['volatility'] * 0.3) + (df['volume_factor'] * 0.3)
    
    # Return the heuristic scores as a Series
    return heuristics_matrix
