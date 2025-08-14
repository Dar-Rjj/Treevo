import Series
    from talib import RSI, EMA
    
    # Calculate 10-day EMA of the volume
    ema_volume = EMA(df['volume'].values, timeperiod=10)
    
    # Calculate 14-day RSI of the close price
    rsi_close = RSI(df['close'].values, timeperiod=14)
    
    # Combine EMA of volume and RSI of close
    heuristics_matrix = Series(ema_volume * (rsi_close / 100), index=df.index)
    
    return heuristics_matrix
