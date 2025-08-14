def heuristics_v2(df):
    # Calculate Daily High-Low Range
    df['daily_range'] = df['high'] - df['low']
    
    # Calculate 21-Day Rolling Sum of High-Low Range and Volume
    df['rolling_sum_range'] = df['daily_range'].rolling(window=21).sum()
    df['rolling_sum_volume'] = df['volume'].rolling(window=21).sum()
    
    # Calculate Breakout Strength
    df['breakout_strength'] = df['daily_range'] / df['rolling_sum_range']
    
    # Calculate Intraday Close-to-Open Delta
    df['intraday_delta'] = df['close'] - df['open']
    
    # Formulate Intraday Reversal Score
    df['intraday_reversal_score'] = (df['intraday_delta'] / df['daily_range']) * df['volume']
    
    # Calculate Price Change
    df['price_change'] = df['close'].pct_change()
    
    # Multiply Price Change by Volume
    df['price_change_vol_weighted'] = df['price_change'] * df['volume']
    
    # Combine Indicators for Breakout and Intraday Reversal
    df['combined_breakout_intraday'] = df['breakout_strength'] + df['price_change_vol_weighted'] + df['intraday_reversal_score']
    
    # Calculate Volume Weighted Momentum
    df['vol_weighted_momentum'] = df['price_change'] * df['volume']
    
    # Combine Breakout and Volume Weighted Momentum
    df['combined_breakout_vol_momentum'] = df['breakout_strength'] + df['vol_weighted_momentum']
    
    # Calculate Price Momentum
    df['7_day_return'] = df['close'].pct_change(7)
