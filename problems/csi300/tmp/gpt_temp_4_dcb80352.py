import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Multi-timeframe momentum blend with regime detection using volume acceleration
    and dynamic quantile thresholds. Combines ultra-short and medium-term signals
    with volatility normalization for robust alpha generation.
    """
    
    # Ultra-short momentum (1-period)
    ultra_short_mom = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Medium-term momentum (5-period)
    medium_mom = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    
    # Volatility estimation using high-low range (5-day rolling)
    volatility = (df['high'] - df['low']).rolling(window=5).mean()
    
    # Volume acceleration (rate of change in volume momentum)
    vol_roc = (df['volume'] - df['volume'].shift(3)) / (df['volume'].shift(3) + 1e-7)
    vol_acceleration = vol_roc - vol_roc.shift(2)
    
    # Regime detection using price and volume dynamics
    price_trend = df['close'].rolling(window=10).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1)
    vol_regime = df['volume'].rolling(window=10).apply(lambda x: 1 if x.mean() > x.median() else -1)
    
    # Momentum blend with regime weighting
    momentum_blend = (0.6 * ultra_short_mom + 0.4 * medium_mom) * (1 + 0.2 * price_trend)
    
    # Volatility-normalized momentum
    vol_normalized_mom = momentum_blend / (volatility + 1e-7)
    
    # Dynamic quantile thresholds for volume acceleration
    vol_accel_upper = vol_acceleration.rolling(window=20).quantile(0.7)
    vol_accel_lower = vol_acceleration.rolling(window=20).quantile(0.3)
    
    # Volume regime signals
    high_vol_accel = vol_acceleration > vol_accel_upper
    low_vol_accel = vol_acceleration < vol_accel_lower
    
    # Core factor construction
    factor = vol_normalized_mom.copy()
    
    # Volume acceleration amplification
    factor = factor.mask(high_vol_accel & (momentum_blend > 0), 
                        vol_normalized_mom * (1 + 0.3 * vol_regime))
    factor = factor.mask(low_vol_accel & (momentum_blend < 0), 
                        vol_normalized_mom * (1 - 0.3 * vol_regime))
    
    # Mean reversion signals in high volatility regimes
    high_vol_regime = volatility > volatility.rolling(window=20).quantile(0.7)
    extreme_moves = abs(momentum_blend) > momentum_blend.rolling(window=20).quantile(0.8)
    
    factor = factor.mask(high_vol_regime & extreme_moves, 
                        -vol_normalized_mom * 0.8)
    
    # Volume confirmation in trending regimes
    strong_trend = abs(momentum_blend) > momentum_blend.rolling(window=20).quantile(0.6)
    vol_confirmation = df['volume'] > df['volume'].rolling(window=10).quantile(0.7)
    
    factor = factor.mask(strong_trend & vol_confirmation, 
                        vol_normalized_mom * 1.2)
    
    return factor
