def heuristics_v2(df):
    # Chaikin Money Flow (CMF)
    money_flow_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    money_flow_volume = money_flow_multiplier * df['volume']
    cmf = money_flow_volume.rolling(window=30).sum() / df['volume'].rolling(window=30).sum()
    
    # Average True Range (ATR)
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift())
    tr3 = abs(df['low'] - df['close'].shift())
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=14).mean()
    
    # Cube root of Closing Price to ATR Ratio
    close_to_atr_ratio_cbrt = np.cbrt(df['close'] / atr)
    
    # Composite heuristic matrix
    heuristics_matrix = 0.6 * cmf + 0.4 * close_to_atr_ratio_cbrt
    
    return heuristics_matrix
