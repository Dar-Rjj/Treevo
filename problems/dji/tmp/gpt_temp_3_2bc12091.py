def heuristics_v2(df):
    df['price_momentum'] = df['close'].pct_change(periods=14)
    df['volume_macd'], _, _ = talib.MACD(df['volume'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['true_range'] = df[['high', 'low', 'close']].apply(lambda x: max(x['high'] - x['low'], abs(x['high'] - x['close'].shift(1)), abs(x['low'] - x['close'].shift(1))), axis=1)
    df['sma_true_range'] = df['true_range'].rolling(window=5).mean()
    heuristics_matrix = (df['price_momentum'] * df['volume_macd']) / df['sma_true_range']
    return heuristics_matrix
