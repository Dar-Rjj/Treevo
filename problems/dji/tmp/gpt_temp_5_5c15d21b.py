def heuristics_v2(df):
    period_roc = 10
    period_wma = 14
    
    # Calculate ROC
    roc = df['close'].pct_change(periods=period_roc)
    
    # Calculate WMA
    weights = pd.Series(range(1, period_wma + 1))
    wma = df['volume'].rolling(window=period_wma).apply(lambda x: (x * weights).sum() / weights.sum(), raw=True)
    
    # Combine ROC and WMA
    heuristics_matrix = (roc * 0.5) + (wma * 0.5)
    
    return heuristics_matrix
