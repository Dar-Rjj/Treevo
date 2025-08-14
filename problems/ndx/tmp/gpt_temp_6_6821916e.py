def heuristics_v2(df):
    # Calculate the True Range (TR)
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift())
    tr3 = abs(df['low'] - df['close'].shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate the Average True Range (ATR) over a 10-day period
    atr = tr.rolling(window=10).mean()
    
    # Calculate the Rate of Change (ROC) over a 14-day period
    roc = (df['close'] - df['close'].shift(14)) / df['close'].shift(14) * 100
    
    # Compute the heuristic score as a composite of ATR and ROC
    heuristics_matrix = atr + roc
    
    return heuristics_matrix
