def heuristics_v2(df):
    # Chaikin Money Flow (CMF)
    money_flow_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    money_flow_volume = money_flow_multiplier * df['volume']
    cmf = money_flow_volume.rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
    cmf_ema = cmf.ewm(span=20, adjust=False).mean()
    
    # Average True Range (ATR)
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift())
    tr3 = abs(df['low'] - df['close'].shift())
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=14).mean()
    
    # Natural Logarithm of Closing Price to ATR Ratio
    close_to_atr_ratio_log = np.log(df['close'] / atr)
    
    # Composite heuristic matrix
    heuristics_matrix = 0.7 * cmf_ema + 0.3 * close_to_atr_ratio_log
    
    return heuristics_matrix
