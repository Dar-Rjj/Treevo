def heuristics_v2(df):
    # Calculate Short-Term Momentum
    df['5_day_ma_close'] = df['close'].rolling(window=5).mean()
    df['short_term_momentum'] = df['close'] - df['5_day_ma_close']
    
    # Calculate Long-Term Momentum
    df['30_day_ma_close'] = df['close'].rolling(window=30).mean()
    df['long_term_momentum'] = df['close'] - df['30_day_ma_close']
    
    # Determine Relative Strength Score
    df['relative_strength_score'] = df['short_term_momentum'] / df['long_term_momentum']
    
    # Calculate Intraday Volatility
    df['intraday_volatility'] = (df['high'] - df['low']) / df['close']
    
    # Combine Momentum and Intraday Volatility
    df['momentum_volatility_composite'] = df['relative_strength_score'] * df['intraday_volatility']
    
    # Compute Short-Term Volume Trend
    df['5_day_ma_volume'] = df['volume'].rolling(window=5).mean()
    
    # Compute Long-Term Volume Trend
    df['30_day_ma_volume'] = df['volume'].rolling(window=30).mean()
    
    # Determine Volume Ratio
    df['volume_ratio'] = df['5_day_ma_volume'] / df['30_day_ma_volume']
    
    # Assign Volume Ratio Score
    df['volume_ratio_score'] = df['volume_ratio'].apply(lambda x: 1 if x > 1 else 0)
    
    # Combine Scores
    df['combined_scores'] = df['relative_strength_score'] * df['volume_ratio_score']
    
    # Adjust Momentum-Volatility Composite
    df['adjusted_momentum_volatility_composite'] = df['momentum_volatility_composite'] * df['combined_scores']
    
    # Calculate High-Low Difference
    df['high_low_diff'] = df['high'] - df['low']
    
    # Adjust High-Low Difference by Open Price
    df['adjusted_high_low_diff'] = df['high_low_diff'] - df['open']
    
    # Calculate Absolute Movement
    df['absolute_movement'] = df['adjusted_high_low_diff'].abs()
    
    # Smooth Over N Days
    df['smoothed_absolute_movement'] = df['absolute_movement'].rolling(window=10).mean()
    
    # Calculate Intraday Open-Close Return
    df['intraday_open_close_return'] = df['close'] / df['open'] - 1
