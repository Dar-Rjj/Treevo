def heuristics_v2(df):
    # Simple Moving Average (SMA)
    df['SMA_5'] = df['close'].rolling(window=5).mean()
    df['SMA_10'] = df['close'].rolling(window=10).mean()
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    
    # Rate of Change (ROC)
    df['ROC_5'] = (df['close'] / df['close'].shift(5) - 1) * 100
    df['ROC_10'] = (df['close'] / df['close'].shift(10) - 1) * 100
    df['ROC_20'] = (df['close'] / df['close'].shift(20) - 1) * 100
    
    # True Range (TR)
    df['TR'] = df[['high', 'low']].sub(df['close'].shift(1), axis=0).abs().max(axis=1)
    
    # Average True Range (ATR)
    df['ATR_14'] = df['TR'].rolling(window=14).mean()
    
    # On-Balance Volume (OBV)
    df['OBV'] = (df['close'] > df['close'].shift(1)).astype(int) * df['volume'] - (df['close'] < df['close'].shift(1)).astype(int) * df['volume']
    df['OBV'] = df['OBV'].cumsum()
    
    # Chaikin Money Flow (CMF)
    mf_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    mf_volume = mf_multiplier * df['volume']
    df['CMF_20'] = mf_volume.rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
    
    # Directional Movement Index (DMI)
    df['DM_pos'] = df['high'].diff()
    df['DM_neg'] = -df['low'].diff()
    df.loc[df['DM_pos'] < 0, 'DM_pos'] = 0
    df.loc[df['DM_neg'] < 0, 'DM_neg'] = 0
    df.loc[df['DM_pos'] < df['DM_neg'], 'DM_pos'] = 0
    df.loc[df['DM_neg'] < df['DM_pos'], 'DM_neg'] = 0
    
    TR_14 = df['TR'].rolling(window=14).sum()
    DM_pos_14 = df['DM_pos'].rolling(window=14).sum()
    DM_neg_14 = df['DM_neg'].rolling(window=14).sum()
    
    df['+DI'] = (DM_pos_14 / TR_14) * 100
    df['-DI'] = (DM_neg_14 / TR_14) * 100
    df['DX'] = (abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])) * 100
    df['ADX'] = df['DX'].rolling(window=14).mean()
    
    # Doji Pattern
    doji_threshold = 0.1
    df['Doji'] = ((df['high'] - df['low']) / (df['close'] - df['open'])).apply(lambda x: 1 if x > doji_threshold else 0)
    
    # Engulfing Pattern
    df['Bullish_Engulfing'] = ((df['close'].shift(1) < df['open'].shift(1)) & 
                                (df['close'] > df['open']) & 
                                (df['close'] > df['open'].shift(1)) & 
                                (df['open'] < df['close'].shift(1))).astype(int)
    df['Bearish_Engulfing'] = ((df['close'].shift(1) > df['open'].shift(1)) & 
                                (df['close'] < df['open']) & 
                                (df['close'] < df['open'].shift(1)) & 
                                (df['open'] > df['close'].shift(1))).astype(int)
    
    # Composite Indicator
    df['Composite_Factor'] = (df['SMA_20'] + df['ROC_20'] + df['ATR_14'] + df['OBV'] + df['+DI'] - df['-DI']) / 5
    
    return df['Composite_F
