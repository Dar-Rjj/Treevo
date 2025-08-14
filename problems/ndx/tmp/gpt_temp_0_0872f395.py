def heuristics_v2(df):
    # Calculate 5-day and 20-day EWMA of close prices
    df['ewma_5'] = df['close'].ewm(span=5).mean()
    df['ewma_20'] = df['close'].ewm(span=20).mean()

    # Calculate 5-day and 20-day rolling total volume
    df['volume_5'] = df['volume'].rolling(window=5).sum()
    df['volume_20'] = df['volume'].rolling(window=20).sum()

    # Calculate 5-day and 20-day VWAP
    df['vwap_5'] = (df['close'] * df['volume']).rolling(window=5).sum() / df['volume_5']
    df['vwap_20'] = (df['close'] * df['volume']).rolling(window=20).sum() / df['volume_20']

    # Calculate 5-day and 20-day ATR
    df['tr'] = df[['high', 'low', 'close']].max(axis=1) - df[['high', 'low', 'close']].min(axis=1)
    df['atr_5'] = df['tr'].rolling(window=5).mean()
    df['atr_20'] = df['tr'].rolling(window=20).mean()

    # Calculate 5-day and 20-day ROC in volume
    df['roc_volume_5'] = df['volume'].pct_change(periods=5)
    df['roc_volume_20'] = df['volume'].pct_change(periods=20)

    # Determine if current day's volume is higher than 5-day and 20-day average volume
    df['volume_higher_5'] = df['volume'] > df['volume_5'] / 5
    df['volume_higher_20'] = df['volume'] > df['volume_20'] / 20

    # Calculate 10-day and 30-day price momentum
    df['price_momentum_10'] = (df['high'] - df['low']).rolling(window=10).sum()
    df['price_momentum_30'] = (df['high'] - df['low']).rolling(window=30).sum()

    # Calculate 10-day and 30-day ROC of close prices
    df['roc_close_10'] = df['close'].pct_change(periods=10)
    df['roc_close_30'] = df['close'].pct_change(periods=30)

    # Calculate 14-day RSI using high and low prices
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # Generate intraday volatility metric
    df['intraday_volatility'] = (df['high'] - df['low']) / df['volume']

    # Rolling 14-day high-to-low range sum
    df['high_low_range_sum_14'] = (df['high'] - df['low']).rolling(window=14).sum()

    # Difference between consecutive VWAP
    df['vwap_diff'] = df['vwap_20'] - df['vwap_5'].shift(1)

    # Calculate 20-day EMA of VWAP
    df['vwap_ema_20'] = df['vwap_20'].ewm(span=20).mean()

    # Calculate 10-day standard deviation of daily returns
