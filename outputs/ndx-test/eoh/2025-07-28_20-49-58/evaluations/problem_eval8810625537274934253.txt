def heuristics_v2(df):
    # Calculate the percentage change in closing prices
    pct_change_close = df['close'].pct_change()
    
    # Calculate the True Range
    tr = df[['high', 'low', 'close']].apply(lambda x: max(x['high'] - x['low'], abs(x['high'] - x['close'].shift(1)), abs(x['low'] - x['close'].shift(1))), axis=1)
    
    # Calculate the Average True Range (ATR) over 21 days
    atr_21 = tr.rolling(window=21).mean()
    
    # Compute the alpha factor as the ratio of the percentage change in closing price to the ATR
    heuristics_matrix = (pct_change_close / atr_21).dropna()
    
    return heuristics_matrix
