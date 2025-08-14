def heuristics_v2(df):
    window = 14  # Period for ATR and SMA calculation
    
    # Calculate the True Range
    df['tr'] = df[['high', 'low', 'close']].diff().abs().max(axis=1)
    
    # Calculate Average True Range (ATR)
    atr = df['tr'].rolling(window).mean()
    
    # Calculate Simple Moving Average (SMA) of the closing price
    sma_close = df['close'].rolling(window).mean()
    
    # Calculate the deviation of the close price from its SMA
    deviation = df['close'] - sma_close
    
    # Adjust the deviation by the ATR to get the Heuristics Matrix
    heuristics_matrix = pd.Series(deviation / atr, name='HeuristicFactor')
    
    return heuristics_matrix
