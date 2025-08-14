def heuristics_v2(df):
    # Calculate Earnings Yield (E/P)
    earnings_yield = 1 / (df['close'] / df['earnings'])
    
    # Calculate Price-to-Earnings Ratio
    pe_ratio = df['close'] / df['earnings']
    
    # Calculate On-Balance Volume (OBV)
    obv = ((df['close'] - df['close'].shift(1)) > 0).astype(int) * df['volume']
    obv = obv.cumsum()
    
    # Composite heuristics factor
    heuristics_matrix = earnings_yield - pe_ratio + obv
    
    return heuristics_matrix
