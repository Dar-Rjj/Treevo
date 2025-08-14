def heuristics_v2(df):
    # Calculate the Rate of Change (ROC) over a 20-day window
    roc = df['close'].pct_change(periods=20)
    
    # Calculate the True Range (TR)
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift())
    tr3 = abs(df['low'] - df['close'].shift())
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate the weighted average of the True Range over 5, 20, and 60 days
    wtr_short = true_range.rolling(window=5).mean()
    wtr_mid = true_range.rolling(window=20).mean()
    wtr_long = true_range.rolling(window=60).mean()
    wtr_avg = 0.4 * wtr_short + 0.3 * wtr_mid + 0.3 * wtr_long
    
    # Combine ROC and weighted average True Range
    heuristics_matrix = roc + wtr_avg
    
    return heuristics_matrix
