def heuristics_v2(df, n=20):
    # Calculate the difference between today's close and the average of the last n days' closes (price trend)
    df['avg_close_n_days'] = df['close'].rolling(window=n).mean()
    df['price_trend'] = df['close'] - df['avg_close_n_days']

    # Identify if the closing price is higher than the opening price and if it has been increasing over the past n days (bullish momentum)
    df['bullish_momentum'] = (df['close'] > df['open']).astype(int) * (df['close'] > df['close'].shift(1)).rolling(window=n).sum()

    # Measure the distance between the closing price and the highest price of the day, and compare it to the average of the last n days (upper shadow indicator)
    df['upper_shadow'] = df['high'] - df['close']
    df['avg_upper_shadow_n_days'] = df['upper_shadow'].rolling(window=n).mean()
    df['upper_shadow_indicator'] = df['upper_shadow'] / df['avg_upper_shadow_n_days']

    # Measure the distance between the closing price and the lowest price of the day, and compare it to the average of the last n days (lower shadow indicator)
    df['lower_shadow'] = df['close'] - df['low']
    df['avg_lower_shadow_n_days'] = df['lower_shadow'].rolling(window=n).mean()
    df['lower_shadow_indicator'] = df['lower_shadow'] / df['avg_lower_shadow_n_days']

    # Compare today's volume to the maximum of the last n days' volumes (volume spike)
    df['max_volume_n_days'] = df['volume'].rolling(window=n).max()
    df['volume_spike'] = df['volume'] / df['max_volume_n_days']

    # Assess the correlation between volume and the change in closing prices over the past n days (volume-price relationship)
    df['price_change'] = df['close'].pct_change()
    df['volume_price_correlation'] = df['volume'].rolling(window=n).corr(df['price_change'])

    # Consider high volume with a positive price change and if this pattern has occurred in the past n days (strength confirmation)
    df['strength_confirmation'] = ((df['volume'] > df['volume'].rolling(window=n).quantile(0.75)) & (df['price_change'] > 0)).rolling(window=n).sum()

    # Consider high volume with a negative price change and if this pattern has occurred in the past n days (weakness confirmation)
    df['weakness_confirmation'] = ((df['volume'] > df['volume'].rolling(window=n).quantile(0.75)) & (df['price_change'] < 0)).rolling(window=n).sum()

    # Check if there is a gap up or down from the previous day's close, and if such gaps have occurred in the past n days (gap frequency)
    df['gap_up'] = (df['open'] > df['close'].shift(1)).astype(int)
    df['gap_down'] = (df['open'] < df['close'].shift(1)).astype(int)
    df['gap_frequency'] = df['gap_up'].rolling(window=n).sum() + df['gap_down'].rolling(window=n).sum()

    # Determine if the body size indicates strong buying or selling pressure, and if this pattern has been consistent over the past n days (body length trend)
    df['body_size'] = abs(df['close'] - df['open'])
    df['avg_body_size_n_days'] = df['body_size'].rolling(window=n).mean()
    df['body_length_trend'] = (df['body_size'] > df['avg_body_size_n_days']).astype(int)

    # Assess the meaning of long or short wicks relative to the body, and if the wick length has been increasing or decreasing over the past n days (wick trend)
    df['wick_length'] = df['high'] - df['low']
    df['avg_wick_length_n_days'] = df['wick_length'].rolling(window=n).mean()
    df['wick_trend'] = (df['wick_length'] > df['avg_wick_length_n_days']).astype(int)

    # Combine all factors into a single alpha factor
    df['alpha_factor'] = (
        df['price_trend'] +
        df['bullish_momentum'] +
        df['upper_shadow_indicator'] -
        df['lower_shadow_indicator'] +
        df['volume_spike'] +
        df['volume_price_correlation'] +
        df['strength_confirmation'] -
        df['weakness_confirmation'] +
        df['gap_frequency'] +
        df['body_length_trend'] +
        df['wick_trend']
    )

    return df['
