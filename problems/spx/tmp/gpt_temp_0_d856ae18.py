def heuristics(df):
    # Compute the 7-day and 21-day simple moving averages (SMA) for the closing prices
    df['SMA_7'] = df['close'].rolling(window=7).mean()
    df['SMA_21'] = df['close'].rolling(window=21).mean()
    
    # Create an alpha factor as the difference between the 7-day SMA and the 21-day SMA
    df['alpha_factor_sma_diff'] = df['SMA_7'] - df['SMA_21']
    
    # Define a 30-day price return as the percentage change in the closing price from 30 days ago
