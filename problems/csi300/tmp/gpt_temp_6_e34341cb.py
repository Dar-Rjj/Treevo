def heuristics_v2(df):
    # Calculate SMAs
    df['SMA_5'] = df['close'].rolling(window=5).mean()
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_60'] = df['close'].rolling(window=60).mean()
    
    # Price-to-moving-average ratios
    df['PMA_5'] = df['close'] / df['SMA_5']
    df['PMA_20'] = df['close'] / df['SMA_20']
    df['PMA_60'] = df['close'] / df['SMA_60']
    
    # Daily price range
    df['daily_range'] = df['high'] - df['low']
    
    # On-balance volume (OBV)
    df['price_change'] = df['close'].diff()
    df['direction'] = df['price_change'].apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
    df['OBV'] = (df['volume'] * df['direction']).cumsum()
    
    # Volume-price trend (VPT)
    df['VPT'] = ((df['close'].pct_change() * df['volume']).fillna(0)).cumsum()
    
    # High-low spread
    df['high_low_spread'] = df['high'] - df['low']
    df['HL_MA_10'] = df['high_low_spread'].rolling(window=10).mean()
    
    # Close higher or lower than open
    df['close_higher_than_open'] = df['close'] > df['open']
    df['close_above_open_10_days'] = df['close_higher_than_open'].rolling(window=10).sum()
    
    # Cumulative returns
