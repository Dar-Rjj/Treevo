def heuristics_v2(df):
    # Calculate short-term price momentum
    df['short_momentum'] = df['close'] - df['close'].rolling(window=5).mean()
    
    # Calculate medium-term price momentum
    df['medium_momentum'] = df['close'] - df['close'].rolling(window=20).mean()
    
    # Calculate long-term price momentum
    df['long_momentum'] = df['close'] - df['close'].rolling(window=60).mean()
    
    # Analyze recent volatility
    df['returns'] = df['close'].pct_change()
