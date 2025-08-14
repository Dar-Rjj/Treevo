def heuristics_v2(df):
    # Calculate the 20-day and 100-day exponential moving averages of the close price
    ema_20 = df['close'].ewm(span=20, adjust=False).mean()
    ema_100 = df['close'].ewm(span=100, adjust=False).mean()
    
    # Calculate the ratio between the EMAs
    ema_ratio = ema_20 / ema_100
    
    # Calculate the True Range
    df['TrueRange'] = df[['high', 'low']].apply(lambda x: max(x['high'], x.shift(1)['close']) - min(x['low'], x.shift(1)['close']), axis=1)
    
    # Calculate the 30-day average true range (ATR)
    atr_30 = df['TrueRange'].rolling(window=30).mean()
    
    # Generate the heuristic matrix by multiplying the EMA ratio with the ATR
    heuristics_matrix = ema_ratio * atr_30
    
    return heuristics_matrix
