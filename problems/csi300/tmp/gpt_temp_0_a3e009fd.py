def heuristics_v2(df):
    # Momentum Indicators
    df['SMA_5'] = df['close'].rolling(window=5).mean()
    df['SMA_10'] = df['close'].rolling(window=10).mean()
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['EMA_5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['EMA_10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ROC_5'] = (df['close'] / df['close'].shift(5) - 1) * 100
    df['ROC_10'] = (df['close'] / df['close'].shift(10) - 1) * 100
    df['ROC_20'] = (df['close'] / df['close'].shift(20) - 1) * 100

    # Volatility Indicators
    df['TR'] = df[['high', 'low', 'close']].diff(axis=1).abs().max(axis=1)
    df['ATR_14'] = df['TR'].rolling(window=14).mean()

    # Volume-Based Indicators
    df['OBV'] = (df['close'] > df['close'].shift(1)).astype(int) * df['volume']
    df['OBV'] -= (df['close'] < df['close'].shift(1)).astype(int) * df['volume']
    df['OBV'] = df['OBV'].cumsum()
    
    def cmf(high, low, close, volume, period):
        mf_multiplier = ((close - low) - (high - close)) / (high - low)
        mf_volume = mf_multiplier * volume
        cmf = mf_volume.rolling(window=period).sum() / volume.rolling(window=period).sum()
        return cmf
