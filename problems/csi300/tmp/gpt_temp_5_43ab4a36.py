def heuristics_v2(df):
    # Simple Moving Averages (SMA) and VWAP
    df['5_day_SMA'] = df['close'].rolling(window=5).mean()
    df['20_day_SMA'] = df['close'].rolling(window=20).mean()
    df['SMA_diff'] = df['5_day_SMA'] - df['20_day_SMA']
    df['5_day_SMA_roc'] = df['5_day_SMA'].pct_change(periods=5)
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['VWAP'] = (df['typical_price'] * df['volume']).cumsum() / df['volume'].cumsum()
    df['VWAP_diff_close'] = df['VWAP'] - df['close']

    # Raw Returns
    df['daily_return'] = df['close'].pct_change()
