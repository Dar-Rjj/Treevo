import pandas as pd
    
    # Momentum factor: 10-day simple moving average over 30-day simple moving average
    df['SMA_10'] = df['close'].rolling(window=10).mean()
    df['SMA_30'] = df['close'].rolling(window=30).mean()
    df['momentum'] = df['SMA_10'] / df['SMA_30']
    
    # Mean Reversion factor: (Close - 50-day SMA) / 50-day Standard Deviation
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    df['STD_50'] = df['close'].rolling(window=50).std()
    df['mean_reversion'] = (df['close'] - df['SMA_50']) / df['STD_50']
    
    # Volatility factor: 20-day standard deviation of log returns
    df['log_ret'] = (df['close'] / df.shift(1)['close']).apply(lambda x: np.log(x))
    df['volatility'] = df['log_ret'].rolling(window=20).std()
    
    # Composite factor: combination of momentum, mean reversion, and volatility
    df['composite_factor'] = df['momentum'] - df['mean_reversion'] + df['volatility']
    
    heuristics_matrix = df['composite_factor']
    
    return heuristics_matrix
