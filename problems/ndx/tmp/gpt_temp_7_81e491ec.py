def heuristics_v2(df):
    # Calculate the 15-day and 30-day exponential moving averages of the close price
    ema_15 = df['close'].ewm(span=15, adjust=False).mean()
    ema_30 = df['close'].ewm(span=30, adjust=False).mean()
    
    # Calculate the ratio between the EMAs
    ema_ratio = ema_15 / ema_30
    
    # Calculate the True Range
    df['TrueRange'] = df[['high', 'low', 'close']].apply(lambda x: max(x['high'], x['close'].shift(1)) - min(x['low'], x['close'].shift(1)), axis=1)
    
    # Calculate the 20-day average true range
    atr_20 = df['TrueRange'].rolling(window=20).mean()
    
    # Generate the heuristic matrix by multiplying the EMA ratio with the ATR
    heuristics_matrix = ema_ratio * atr_20
    
    return heuristics_matrix
