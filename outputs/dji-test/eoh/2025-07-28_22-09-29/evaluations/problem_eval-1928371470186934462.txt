def heuristics_v2(df):
    df['up_move'] = df['high'].diff()
    df['down_move'] = df['low'].diff().abs()
    
    df['+DM'] = 0
    df['-DM'] = 0
    df.loc[(df['up_move'] > df['down_move']) & (df['up_move'] > 0), '+DM'] = df['up_move']
    df.loc[(df['down_move'] > df['up_move']) & (df['down_move'] > 0), '-DM'] = df['down_move']
    
    df['+DM21'] = df['+DM'].rolling(window=21).mean()
    df['-DM21'] = df['-DM'].rolling(window=21).mean()
    
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift())
    tr3 = abs(df['low'] - df['close'].shift())
    df['TrueRange'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    df['ATR21'] = df['TrueRange'].rolling(window=21).mean()
    
    df['+DI'] = (df['+DM21'] / df['ATR21']) * 100
    df['-DI'] = (df['-DM21'] / df['ATR21']) * 100
    
    df['DX'] = (abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])) * 100
    
    df['CumulativeVolume'] = df['volume'].rolling(window=21).sum()
    
    heuristics_matrix = df['DX'] * (df['CumulativeVolume'] / df['ATR21'])
    return heuristics_matrix
