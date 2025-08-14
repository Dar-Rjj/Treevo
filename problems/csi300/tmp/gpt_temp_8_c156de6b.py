def heuristics(df):
    # Calculate Daily Price Momentum
    df['price_momentum'] = df['close'].diff(10)
    
    # Calculate Volume Surprise
    df['volume_ma_10'] = df['volume'].rolling(window=10).mean()
    df['volume_surprise'] = df['volume'] - df['volume_ma_10']
    
    # Combine Price Momentum and Volume Surprise
    df['price_mom_vol_surp'] = df['price_momentum'] * df['volume_surprise']
    
    # Calculate High-Low Price Difference
    df['high_low_diff'] = df['high'] - df['low']
    
    # Compute Volume Influence Ratio
    df['upward_volume'] = df.apply(lambda x: x['volume'] if x['close'] > x['open'] else 0, axis=1)
    df['downward_volume'] = df.apply(lambda x: x['volume'] if x['close'] < x['open'] else 0, axis=1)
    df['upward_volume_sum'] = df['upward_volume'].rolling(window=10).sum()
    df['downward_volume_sum'] = df['downward_volume'].rolling(window=10).sum()
    df['volume_influence_ratio'] = df['upward_volume_sum'] / df['downward_volume_sum'].replace(0, 1)
    
    # Intermediate Alpha Factor Synthesis
    df['intermediate_alpha'] = df['price_mom_vol_surp'] * df['high_low_diff']
    
    # Calculate Intraday Return
    df['intraday_return'] = (df['close'] - df['open']) / df['open']
