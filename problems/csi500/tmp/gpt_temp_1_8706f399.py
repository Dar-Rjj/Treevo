import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Adaptive Momentum with Volume-Price Synchronization and Dynamic Volatility Scaling
    # Captures momentum strength validated by volume synchronization and adjusted for regime-specific volatility
    # Positive values indicate strong momentum with volume confirmation in appropriate volatility regimes
    
    # Adaptive momentum using multiple time horizons (5-day and 10-day)
    momentum_5 = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    momentum_10 = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
    
    # Volume synchronization factor - measures if volume supports price movement direction
    volume_avg_5 = df['volume'].rolling(5).mean()
    volume_avg_10 = df['volume'].rolling(10).mean()
    volume_sync_5 = (df['volume'] - volume_avg_5) / volume_avg_5
    volume_sync_10 = (df['volume'] - volume_avg_10) / volume_avg_10
    
    # Dynamic volatility regime detection using rolling percentiles
    high_low_range = df['high'] - df['low']
    true_range = pd.concat([
        high_low_range,
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    
    vol_regime_20 = true_range.rolling(20).apply(lambda x: (x.iloc[-1] - x.mean()) / x.std())
    vol_regime_5 = true_range.rolling(5).apply(lambda x: (x.iloc[-1] - x.mean()) / x.std())
    
    # Price efficiency factor - measures how efficiently price moves within daily range
    daily_efficiency = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-7)
    
    # Combine factors with regime-aware weighting
    short_term_component = momentum_5 * volume_sync_5 * (1 - abs(vol_regime_5))
    long_term_component = momentum_10 * volume_sync_10 * (1 - abs(vol_regime_20))
    efficiency_component = daily_efficiency * (df['amount'] / df['volume']) / (df['close'] + 1e-7)
    
    # Final alpha factor with dynamic regime adjustment
    alpha_factor = (short_term_component + 0.7 * long_term_component) * efficiency_component
    
    return alpha_factor
