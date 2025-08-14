def heuristics_v2(df):
    # Simple Moving Averages (SMA)
    df['SMA_5'] = df['close'].rolling(window=5).mean()
    df['SMA_10'] = df['close'].rolling(window=10).mean()
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    
    df['VOLUME_SMA_5'] = df['volume'].rolling(window=5).mean()
    df['VOLUME_SMA_10'] = df['volume'].rolling(window=10).mean()
    df['VOLUME_SMA_20'] = df['volume'].rolling(window=20).mean()
    
    # Momentum Factors
    df['MOMENTUM_1D'] = df['close'].pct_change(periods=1)
    df['MOMENTUM_5D'] = df['close'].pct_change(periods=5)
    df['MOMENTUM_10D'] = df['close'].pct_change(periods=10)
    df['MOMENTUM_20D'] = df['close'].pct_change(periods=20)
    
    # Volatility
    df['TR'] = df[['high', 'low', 'close']].apply(lambda x: max(x[0] - x[1], abs(x[0] - x[2].shift(1)), abs(x[1] - x[2].shift(1))), axis=1)
    df['ATR_5'] = df['TR'].rolling(window=5).mean()
    df['ATR_10'] = df['TR'].rolling(window=10).mean()
    df['ATR_20'] = df['TR'].rolling(window=20).mean()
    
    # Pattern-based Factors
    df['DOJI'] = (df['open'] - df['close']).abs() < 0.005 * (df['high'] - df['low'])
    df['BULLISH_ENGULFING'] = (df['open'] < df['close']) & (df['open'].shift(1) > df['close'].shift(1)) & (df['close'] > df['open'].shift(1))
    df['BEARISH_ENGULFING'] = (df['open'] > df['close']) & (df['open'].shift(1) < df['close'].shift(1)) & (df['close'] < df['open'].shift(1))
    
    # Volume-Weighted Average Price (VWAP)
    df['VWAP'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4 * df['volume']
    df['VWAP'] = df['VWAP'].cumsum() / df['volume'].cumsum()
    df['VWAP_5'] = df['VWAP'].rolling(window=5).mean()
    df['VWAP_10'] = df['VWAP'].rolling(window=10).mean()
    df['VWAP_20'] = df['VWAP'].rolling(window=20).mean()
    
    # Relative Strength (RS)
    # Assuming benchmark returns are not provided, we use a synthetic benchmark
