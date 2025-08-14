def heuristics_v2(df):
    # Calculate the positive and negative directional movements
    df['up_move'] = df['high'].diff()
    df['down_move'] = df['low'].diff().abs()
    
    # Calculate +DM and -DM
    df['+DM'] = 0
    df['-DM'] = 0
    df.loc[(df['up_move'] > df['down_move']) & (df['up_move'] > 0), '+DM'] = df['up_move']
    df.loc[(df['down_move'] > df['up_move']) & (df['down_move'] > 0), '-DM'] = df['down_move']
    
    # Calculate the smoothed +DM7 and -DM7
    df['+DM7'] = df['+DM'].rolling(window=7).mean()
    df['-DM7'] = df['-DM'].rolling(window=7).mean()
    
    # Calculate the True Range
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift())
    tr3 = abs(df['low'] - df['close'].shift())
    df['TrueRange'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate the 21-day Average True Range (ATR)
    df['ATR21'] = df['TrueRange'].rolling(window=21).mean()
    
    # Calculate the Directional Indicator (DI)
    df['+DI'] = (df['+DM7'] / df['ATR21']) * 100
    df['-DI'] = (df['-DM7'] / df['ATR21']) * 100
    
    # Calculate the Directional Movement Index (DX)
    df['DX'] = (abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])) * 100
    
    # Calculate the cumulative volume
    df['CumulativeVolume'] = df['volume'].rolling(window=28).sum()
    
    # Combine DX with the ratio of cumulative volume to ATR
    heuristics_matrix = df['DX'] * (df['CumulativeVolume'] / df['ATR21'])
    return heuristics_matrix
