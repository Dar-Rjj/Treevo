def heuristics_v2(df):
    # Intraday Momentum Indicators
    df['intraday_momentum'] = (df['close'] - df['open']) / df['open']
    df['intraday_volatility'] = (df['high'] - df['low']) / df['low']
    
    # Volume and Price Correlation
    df['volume_price_corr'] = df['volume'] * (df['close'] - df['open'])
    df['avg_volume_price_corr_5d'] = df['volume_price_corr'].rolling(window=5).mean()
    
    # Inter-day Movement Analyzers
    df['sma_5d'] = df['close'].rolling(window=5).mean()
    df['sma_10d'] = df['close'].rolling(window=10).mean()
    df['sma_20d'] = df['close'].rolling(window=20).mean()
    df['sma_diff_5_20d'] = df['sma_5d'] - df['sma_20d']
    df['ema_10d'] = df['close'].ewm(span=10, adjust=False).mean()
    
    # Relative Strength Analysis
    df['highest_close_10d'] = df['close'].rolling(window=10).max()
    df['lowest_close_10d'] = df['close'].rolling(window=10).min()
    df['relative_strength'] = df['highest_close_10d'] / df['lowest_close_10d']
    
    # Market Sentiment Measures
    df['up_day'] = (df['close'] > df['open']).astype(int)
    df['down_day'] = (df['close'] < df['open']).astype(int)
    df['up_day_count_20d'] = df['up_day'].rolling(window=20).sum()
    df['down_day_count_20d'] = df['down_day'].rolling(window=20).sum()
    df['up_down_ratio'] = df['up_day_count_20d'] / df['down_day_count_20d']
    
    # High/Low Breakout Patterns
    df['high_breakout'] = ((df['high'] > df['high'].shift(1)) & (df['close'] > df['open'])).astype(int)
    df['low_breakout'] = ((df['low'] < df['low'].shift(1)) & (df['close'] < df['open'])).astype(int)
    df['high_low_breakout_ratio'] = df['high_breakout'].rolling(window=20).sum() / df['low_breakout'].rolling(window=20).sum()
    
    # Advanced Volume-Based Metrics
    df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
    df['volume_pressure'] = df['volume_price_corr'].rolling(window=20).sum()
    df['avg_volume_pressure_5d'] = df['volume_price_corr'].rolling(window=5).mean()
    
    # Financial Health Indicators
    # Assuming these columns are present in the DataFrame
    df['profit_margin'] = df['net_income'] / df['revenue']
    df['debt_to_equity'] = df['total_liabilities'] / df['shareholders_equity']
    df['return_on_assets'] = df['net_income'] / df['total_assets']
