def heuristics_v2(df):
    # Calculate Daily Price Change
    df['daily_price_change'] = df['close'] - df['close'].shift(1)
    
    # Calculate Raw Momentum
    n = 5
    df['raw_momentum'] = df['daily_price_change'].rolling(window=n).sum() / n
    
    # Identify Volume Spikes
    m = 20
    df['volume_ma'] = df['volume'].rolling(window=m).mean()
    volume_spike_threshold = 2.0
    df['is_volume_spike'] = (df['volume'] > volume_spike_threshold * df['volume_ma']).astype(int)
    
    # Adjust Momentum by Volume Spike
    momentum_dampening_factor = 0.8
    df['adjusted_momentum'] = df.apply(lambda row: row['raw_momentum'] * momentum_dampening_factor if row['is_volume_spike'] else row['raw_momentum'], axis=1)
    
    # Calculate Intraday High-Low Spread
    df['intraday_high_low_spread'] = df['high'] - df['low']
    
    # Calculate Close-Open Spread
    df['close_open_spread'] = df['close'] - df['open']
    
    # Calculate Intraday Return
    df['intraday_return'] = (df['high'] - df['low']) / df['open']
