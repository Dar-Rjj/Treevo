def heuristics_v2(df):
    # Calculate Intraday Range
    df['intraday_range'] = df['high'] - df['low']
    
    # Adjust Intraday Range by Volume
    ema_span = 10
    df['volume_ema'] = df['volume'].ewm(span=ema_span, adjust=False).mean()
    df['adjusted_volume'] = df['volume'] / df['volume_ema']
    df['adjusted_intraday_range'] = df['intraday_range'] * df['adjusted_volume']
    
    # Normalize by OHLC Volatility
    df['true_range'] = df[['high', 'low', 'close']].max(axis=1) - df[['high', 'low', 'close']].min(axis=1)
    df['ohlc_volatility'] = df['true_range'].rolling(window=20).std()
    df['normalized_intraday_range'] = df['adjusted_intraday_range'] / df['ohlc_volatility']
    
    # Further Adjustment by Close Price Volatility
    df['returns'] = df['close'].pct_change()
