import pandas as pd
    
    # Momentum factor: 12-month price change
    momentum = df['close'].pct_change(12)
    
    # Volatility factor: Standard deviation of daily returns over the last 30 days
    daily_returns = df['close'].pct_change()
    volatility = daily_returns.rolling(window=30).std()
    
    # Trading Volume Analysis: Volume relative to its 30-day average
    avg_volume_30 = df['volume'].rolling(window=30).mean()
    volume_factor = df['volume'] / avg_volume_30
    
    # Combine factors into a heuristics matrix
    heuristics_matrix = (momentum + 1/volatility + volume_factor) / 3
    heuristics_matrix = heuristics_matrix.to_frame(name='heuristic')
    
    return heuristics_matrix
