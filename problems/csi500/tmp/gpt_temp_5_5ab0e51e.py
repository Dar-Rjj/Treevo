def heuristics_v2(df, short_sma_period=50, long_sma_period=200, vol_lookback=20, momentum_lookback=10, pct_change_lookback=5, turnover_lookback=30, recent_performance_period=10):
    # Calculate Simple Moving Average (SMA) of Close Prices
    df['SMA'] = df['close'].rolling(window=short_sma_period).mean()
    
    # Compute Volume-Adjusted Volatility
    df['high_low_diff'] = df['high'] - df['low']
    df['vol_weighted_high_low'] = df['high_low_diff'] * df['volume']
    df['vol_adjusted_volatility'] = df['vol_weighted_high_low'].rolling(window=vol_lookback).mean()
    
    # Compute Price Momentum
    df['avg_close'] = df['close'].rolling(window=momentum_lookback).mean()
    df['price_momentum'] = (df['close'] - df['SMA']) / df['avg_close']
    
    # Incorporate Additional Price Change Metrics
    df['pct_change_close'] = df['close'].pct_change(periods=pct_change_lookback)
    df['high_low_range'] = df['high'] - df['low']
    
    # Consider Market Trend Alignment
    df['long_term_SMA'] = df['close'].rolling(window=long_sma_period).mean()
    df['trend_indicator'] = (df['SMA'] > df['long_term_SMA']).astype(int)
    
    # Incorporate Dynamic Liquidity Measures
    df['daily_turnover'] = df['volume'] * df['close']
    df['rolling_avg_turnover'] = df['daily_turnover'].rolling(window=turnover_lookback).mean()
    df['liquidity_weight'] = df['rolling_avg_turnover'] / df['rolling_avg_turnover'].max()
    
    # Evaluate Recent Performance
    df['recent_price_momentum_return'] = df['price_momentum'].pct_change(periods=recent_performance_period)
