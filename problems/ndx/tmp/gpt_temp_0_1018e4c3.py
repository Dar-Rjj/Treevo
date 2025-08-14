def heuristics_v2(df):
    # Volume-Weighted High-Low Price Difference
    df['vol_weighted_high_low'] = (df['high'] - df['low']) * df['volume']
    
    # Daily Price Change
    df['daily_price_change'] = df['close'].diff()
    
    # Volume Impact Factor
    df['volume_impact_factor'] = df['volume'] * df['daily_price_change'].abs()
    
    # Historical High-Low Range and Momentum
    df['last_5_days_vol_weighted_high_low'] = df['vol_weighted_high_low'].rolling(window=5).sum()
    df['momentum_contribution'] = df['last_5_days_vol_weighted_high_low'] * df['daily_price_change']
    df['positive_momentum_contribution'] = df['momentum_contribution'].apply(lambda x: x if x > 0 else 0)
    
    # Market Sentiment Adjustment
    df['avg_high_low_close'] = (df['high'] - df['low']) / df['close']
    df['volatility_threshold'] = df['avg_high_low_close'].rolling(window=5).mean()
    df['market_sentiment_adjustment'] = df.apply(
        lambda row: row['positive_momentum_contribution'] * 1.1 if row['positive_momentum_contribution'] > row['volatility_threshold'] else row['positive_momentum_contribution'] * 0.9, axis=1
    )
    
    # Overnight Sentiment
    df['overnight_return'] = df['open'] / df['close'].shift(1)
