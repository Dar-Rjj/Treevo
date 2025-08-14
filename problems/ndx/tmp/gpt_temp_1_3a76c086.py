def heuristics_v2(df):
    # Calculate the 60-day and 150-day simple moving averages of the close price
    sma_60 = df['close'].rolling(window=60).mean()
    sma_150 = df['close'].rolling(window=150).mean()
    
    # Calculate the ratio between the SMAs
    sma_ratio = sma_60 / sma_150
    
    # Compute the True Range
    df['True Range'] = df[['high', 'low', 'close']].apply(lambda x: max(x['high'], x['close'].shift(1)) - min(x['low'], x['close'].shift(1)), axis=1)
    
    # Calculate the 40-day average true range (ATR)
    atr_40 = df['True Range'].rolling(window=40).mean()
    
    # Generate the heuristic matrix by multiplying the SMA ratio with the ATR
    heuristics_matrix = sma_ratio * atr_40
    
    return heuristics_matrix
