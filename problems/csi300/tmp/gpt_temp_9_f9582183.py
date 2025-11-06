def heuristics_v2(df):
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']
    amount = df['amount']
    
    # Momentum component with acceleration
    ret_5 = close.pct_change(5)
    ret_10 = close.pct_change(10)
    momentum_accel = ret_5 - ret_10.shift(5)
    
    # Volatility regime adjustment
    vol_20 = close.pct_change().rolling(20).std()
    vol_5 = close.pct_change().rolling(5).std()
    vol_regime = vol_5 / vol_20
    
    # Volume-based reversal signal
    vwap = amount / volume
    price_vwap_ratio = close / vwap
    volume_ma_ratio = volume / volume.rolling(20).mean()
    volume_reversal = -price_vwap_ratio * volume_ma_ratio
    
    # Dynamic thresholding for momentum
    mom_threshold = ret_10.rolling(30).std()
    scaled_momentum = momentum_accel / mom_threshold
    
    # Combined factor with interaction terms
    heuristics_matrix = scaled_momentum * vol_regime + volume_reversal
    
    return heuristics_matrix
