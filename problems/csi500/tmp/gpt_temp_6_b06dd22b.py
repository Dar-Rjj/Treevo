def heuristics_v2(df):
    # Calculate Daily High-Low Range
    df['high_low_range'] = df['high'] - df['low']
    
    # 10-Day Cumulative High-Low Ranges
    df['cum_high_low_range_10'] = df['high_low_range'].rolling(window=10).sum()
    
    # Smoothed Intraday and Cumulative Ranges
    df['ema_high_low_range'] = df['high_low_range'].ewm(span=21, adjust=False).mean()
    df['ema_cum_high_low_range_10'] = df['cum_high_low_range_10'].ewm(span=21, adjust=False).mean()
    
    # Derive Smoothed Momentum Scores
    df['momentum_high_low_range'] = df['ema_high_low_range'].diff()
    df['momentum_cum_high_low_range_10'] = df['ema_cum_high_low_range_10'].diff()
    
    # Price Momentum
    df['price_change_10'] = df['close'].diff(periods=10)
    df['daily_price_momentum'] = df['close'].diff()
    df['ema_close_price_trend'] = df['close'].ewm(span=21, adjust=False).mean()
    df['momentum_close_price_trend'] = df['ema_close_price_trend'].diff()
    
    # Classify Volume
    lookback_period = 20
    df['avg_volume'] = df['volume'].rolling(window=lookback_period).mean()
    df['volume_classification'] = (df['volume'] > df['avg_volume']).astype(int)
    
    # Classify Amount
    df['avg_amount'] = df['amount'].rolling(window=lookback_period).mean()
    df['amount_classification'] = (df['amount'] > df['avg_amount']).astype(int)
    
    # Combine Price Momentum with Volume and Amount Classification
    df['weight'] = 0.0
    df.loc[(df['volume_classification'] == 1) & (df['amount_classification'] == 1), 'weight'] = 1.5
    df.loc[(df['volume_classification'] == 1) & (df['amount_classification'] == 0), 'weight'] = 1.0
    df.loc[(df['volume_classification'] == 0) & (df['amount_classification'] == 1), 'weight'] = 1.0
    df.loc[(df['volume_classification'] == 0) & (df['amount_classification'] == 0), 'weight'] = 0.5
    df['weighted_price_momentum'] = df['daily_price_momentum'] * df['weight']
    
    # Volume Trend
    df['volume_trend'] = df['volume'].rolling(window=21).mean()
    df['volume_score'] = df['volume_trend'].diff()
    
    # Volume-Weighted Intraday Move
    df['volume_weighted_intraday_move'] = df['high_low_range'] * (df['volume'] / df['open'])
    
    # Short-Term Momentum Signal
    df['sma_5'] = df['close'].rolling(window=5).mean()
    df['sma_10'] = df['close'].rolling(window=10).mean()
    df['momentum_signal'] = df['sma_5'] - df['sma_10']
    
    # Combined Intraday-Momentum Factor
    df['combined_intraday_momentum'] = df['volume_weighted_intraday_move'] * df['momentum_signal']
    
    # Intraday Return
    df['intraday_return'] = (df['close'] - df['open']) / df['open']
