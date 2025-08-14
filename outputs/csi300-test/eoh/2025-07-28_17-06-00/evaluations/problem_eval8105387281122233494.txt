def heuristics_v2(df):
    # Calculate the true range
    df['true_range'] = df[['high', 'low']].max(axis=1) - df[['high', 'low']].min(axis=1)
    
    # Calculate the directional movement
    df['up_move'] = df['high'] - df['high'].shift(1)
    df['down_move'] = df['low'].shift(1) - df['low']
    
    # Positive and negative directional movement
    df['pos_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
    df['neg_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
    
    # Smoothed true range and directional movement
    window_size = 14
    df['avg_true_range'] = df['true_range'].rolling(window=window_size).mean()
    df['pos_di'] = 100 * (df['pos_dm'].rolling(window=window_size).sum() / df['avg_true_range'])
    df['neg_di'] = 100 * (df['neg_dm'].rolling(window=window_size).sum() / df['avg_true_range'])
    
    # Directional Movement Index (DMI)
    df['dmi'] = df['pos_di'] - df['neg_di']
    
    # Accumulation/Distribution Line
    df['adl'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']) * df['volume']
    df['adl'] = df['adl'].cumsum()
    
    # Volatility factor
    df['volatility'] = df['close'].rolling(window=window_size).std()
    
    # Heuristic formula: DMI + ADL + Volatility
    heuristics_matrix = df['dmi'] + df['adl'] + df['volatility']
    
    return heuristics_matrix
