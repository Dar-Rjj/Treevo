def heuristics_v2(df):
    # Calculate Simple Moving Averages (SMA)
    df['SMA_5'] = df['close'].rolling(window=5).mean()
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_60'] = df['close'].rolling(window=60).mean()

    # Analyze the difference and ratio between various SMAs
    df['SMA_diff_short_medium'] = df['SMA_5'] - df['SMA_20']
    df['SMA_diff_medium_long'] = df['SMA_20'] - df['SMA_60']
    df['SMA_ratio_short_long'] = df['SMA_5'] / df['SMA_60']

    # Examine volume in relation to price movements
    df['volume_weighted_positive'] = df['volume'] * (df['close'] > df['close'].shift(1)).astype(int)
    df['volume_weighted_negative'] = df['volume'] * (df['close'] < df['close'].shift(1)).astype(int)

    # Incorporate range (High - Low) for volatility insight
    df['range'] = df['high'] - df['low']
    df['ATR_14'] = df['range'].rolling(window=14).mean()
    df['range_ratio'] = df['range'] / df['close']

    # Evaluate the relationship between open and close prices
    df['open_close_diff'] = df['close'] - df['open']
    df['up_day'] = (df['close'] > df['open']).astype(int)
    df['down_day'] = (df['close'] < df['open']).astype(int)

    # Investigate high and low prices to detect support and resistance levels
    df['high_low_diff'] = df['high'] - df['low']
    df['touch_high'] = (df['high'] == df['high'].rolling(window=10).max()).astype(int)
    df['touch_low'] = (df['low'] == df['low'].rolling(window=10).min()).astype(int)

    # Integrate trade amount data
    df['amount_weighted_price_change'] = df['amount'] * df['close'].pct_change()
    df['cumulative_amount_20'] = df['amount'].rolling(window=20).sum()

    # Enhance the analysis with additional price and volume derivatives
    df['percentage_change_5'] = df['close'].pct_change(periods=5)
    df['percentage_change_20'] = df['close'].pct_change(periods=20)
    df['percentage_change_60'] = df['close'].pct_change(periods=60)

    # Compute the On-Balance Volume (OBV)
    df['OBV'] = (df['close'] > df['close'].shift(1)).astype(int) * df['volume'] - (df['close'] < df['close'].shift(1)).astype(int) * df['volume']
    df['OBV_5'] = df['OBV'].rolling(window=5).sum()
    df['OBV_20'] = df['OBV'].rolling(window=20).sum()
    df['OBV_60'] = df['OBV'].rolling(window=60).sum()

    # Incorporate the relationship between trading volume and price changes
    df['VWAP'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    df['VWMA_5'] = (df['close'] * df['volume']).rolling(window=5).sum() / df['volume'].rolling(window=5).sum()
    df['VWMA_20'] = (df['close'] * df['volume']).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
    df['VWMA_60'] = (df['close'] * df['volume']).rolling(window=60).sum() / df['volume'].rolling(window=60).sum()

    # Analyze the interaction between open, high, low, and close prices
    df['body_size'] = df['close'] - df['open']
    df['upper_shadow'] = df['high'] - df[['close', 'open']].max(axis=1)
    df['lower_shadow'] = df[['close', 'open']].min(axis=1) - df['low']

    # Evaluate the impact of gaps on future returns
