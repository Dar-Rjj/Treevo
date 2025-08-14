def heuristics_v2(df):
    # Calculate the rate of change (ROC) over 10 periods
    roc = df['close'].pct_change(periods=10)
    
    # Calculate the On-Balance Volume (OBV)
    obv = (np.sign(df['close'].diff()) * df['volume']).cumsum()
    
    # Factor calculation: Weighted sum of ROC and OBV
    heuristics_matrix = 0.6 * roc + 0.4 * obv
    
    return heuristics_matrix
