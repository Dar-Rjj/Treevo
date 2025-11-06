def heuristics_v2(df):
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    amount = df['amount']
    
    # Volatility-normalized momentum acceleration
    ret_8 = close.pct_change(8)
    ret_21 = close.pct_change(21)
    momentum_accel = (ret_8 - ret_21.shift(8))
    vol_21 = close.pct_change().rolling(21).std()
    normalized_momentum = momentum_accel / vol_21.replace(0, 1e-6)
    
    # Liquidity-adjusted mean reversion
    vwap = amount / volume.replace(0, 1e-6)
    price_vwap_deviation = (close - vwap) / close
    volume_rank = volume.rolling(21).apply(lambda x: (x[-1] - x.mean()) / x.std())
    mean_reversion = -price_vwap_deviation * volume_rank.replace(float('inf'), 0).replace(float('-inf'), 0)
    
    # Volume-confirmed trend persistence
    high_21 = high.rolling(21).max()
    low_21 = low.rolling(21).min()
    breakout_level = (close - low_21) / (high_21 - low_21).replace(0, 1e-6)
    volume_trend = volume.rolling(8).mean() / volume.rolling(21).mean().replace(0, 1e-6)
    trend_persistence = breakout_level * volume_trend
    
    # Entropy-weighted fusion
    momentum_entropy = abs(normalized_momentum.rolling(21).std())
    reversion_entropy = abs(mean_reversion.rolling(21).std())
    trend_entropy = abs(trend_persistence.rolling(21).std())
    
    total_entropy = momentum_entropy + reversion_entropy + trend_entropy
    w1 = momentum_entropy / total_entropy.replace(0, 1e-6)
    w2 = reversion_entropy / total_entropy.replace(0, 1e-6)
    w3 = trend_entropy / total_entropy.replace(0, 1e-6)
    
    heuristics_matrix = (normalized_momentum * w1 + 
                        mean_reversion * w2 + 
                        trend_persistence * w3)
    
    return heuristics_matrix
