import pandas as pd
    
    # Calculate simple return
    simple_return = df['close'].pct_change()
    
    # Momentum: 1-month return
    momentum = df['close'].pct_change(21)
    
    # Volatility: standard deviation of the last 21 days' close prices
    volatility = df['close'].rolling(window=21).std()
    
    # Liquidity: volume change ratio
    volume_change = df['volume'].pct_change()
    
    # Heuristic factor: a simple average of momentum, inverse of volatility (for stability), and volume change
    heuristics_factor = (momentum + 1/volatility + volume_change) / 3
    
    return heuristics_matrix
