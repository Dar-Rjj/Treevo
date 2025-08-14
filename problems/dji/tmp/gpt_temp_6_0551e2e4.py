def heuristics_v2(df):
    step1 = df['close'].rolling(window=20).mean()
    step2 = df['volume'].rolling(window=20).mean()
    step3 = (step1 / step2).fillna(0)
    daily_log_return = (df['close'] / df['close'].shift(1)).apply(np.log).fillna(0)
    step4 = daily_log_return.ewm(span=10, adjust=False).std().fillna(0)
    heuristics_matrix = step3 - step4
    
    return heuristics_matrix
