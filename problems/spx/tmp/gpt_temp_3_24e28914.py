def heuristics_v2(df):
    # Calculate 7-day and 21-day simple moving averages of closing prices
    df['SMA_7'] = df['close'].rolling(window=7).mean()
    df['SMA_21'] = df['close'].rolling(window=21).mean()
    
    # Calculate the true range (TR) for each day
    df['TR'] = df[['high', 'low', 'close']].max(axis=1) - df[['high', 'low', 'close']].min(axis=1)
    
    # Compute the average true range (ATR) over 14 days
    df['ATR_14'] = df['TR'].rolling(window=14).mean()
    
    # Calculate daily returns
