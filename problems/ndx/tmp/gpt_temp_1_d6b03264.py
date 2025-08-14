def heuristics_v2(df):
    # On-Balance Volume (OBV)
    obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    obv_ema = obv.ewm(span=20, adjust=False).mean()
    
    # Average True Range (ATR)
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift())
    tr3 = abs(df['low'] - df['close'].shift())
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=14).mean()
    
    # Logarithm of Closing Price to ATR Ratio
    log_close_to_atr_ratio = np.log(df['close'] / atr)
    
    # Composite heuristic matrix
    heuristics_matrix = 0.7 * obv_ema + 0.3 * log_close_to_atr_ratio
    
    return heuristics_matrix
