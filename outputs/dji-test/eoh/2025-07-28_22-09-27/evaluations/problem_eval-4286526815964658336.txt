import pandas as pd
    
    # Momentum Indicator: simple price change over the last 5 days
    df['momentum'] = df['close'].pct_change(5)
    
    # Volatility: standard deviation of daily returns over the last 20 days
    df['volatility'] = df['close'].pct_change().rolling(window=20).std()
    
    # Volume Trend: difference between current volume and 10-day moving average
    df['volume_trend'] = df['volume'] - df['volume'].rolling(window=10).mean()
    
    # Heuristic matrix as output, selecting only the new created columns
    heuristics_matrix = df[['momentum', 'volatility', 'volume_trend']]
    
    return heuristics_matrix
