def heuristics_v2(df):
    close_prices = df['close']
    
    sma_20_close = close_prices.rolling(window=20, min_periods=1).mean()
    sma_7_close = close_prices.rolling(window=7, min_periods=1).mean()
    tr = df[['high', 'low', 'close']].apply(lambda x: max(x['high'] - x['low'], abs(x['high'] - x['close'].shift(1)), abs(x['low'] - x['close'].shift(1))), axis=1)
    atr_14 = tr.rolling(window=14, min_periods=1).mean()
    
    heuristics_matrix = (sma_20_close - sma_7_close) * atr_14
    
    return heuristics_matrix
