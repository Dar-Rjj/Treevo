def heuristics_v2(df):
    # On-Balance Volume (OBV)
    obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    obv_ema = obv.ewm(span=20, adjust=False).mean()
    
    # 30-day Bollinger Bands
    sma_30 = df['close'].rolling(window=30, min_periods=30).mean()
    std_30 = df['close'].rolling(window=30, min_periods=30).std()
    bollinger_width = 2 * std_30
    
    # Ratio of Closing Price to Bollinger Band Width
    close_to_bbw_ratio = df['close'] / bollinger_width
    
    # Composite heuristic matrix
    heuristics_matrix = 0.7 * obv_ema + 0.3 * close_to_bbw_ratio
    
    return heuristics_matrix
