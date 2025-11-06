import pandas as pd
import pandas as pd

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # 10-20 day momentum divergence with regime-aware volatility scaling and volume confirmation
    # Captures medium-term momentum patterns using 10-day vs 20-day returns with directional consistency
    # Employs regime-aware volatility scaling using 30-day rolling volatility percentiles
    # Volume confirmation uses 15-day volume trend persistence and acceleration
    # Higher values indicate strong medium-term momentum divergence with supportive volume dynamics
    
    # 10-day and 20-day momentum for divergence detection
    momentum_10d = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
    momentum_20d = (df['close'] - df['close'].shift(20)) / df['close'].shift(20)
    
    # Momentum divergence: difference between medium-term momentum signals
    momentum_divergence = momentum_10d - momentum_20d
    
    # Directional consistency: sign agreement between recent and medium-term momentum
    momentum_5d = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    directional_consistency = (momentum_5d * momentum_10d > 0).astype(float)
    
    # Enhanced momentum with directional confirmation
    enhanced_momentum = momentum_divergence * (1 + 0.5 * directional_consistency)
    
    # 30-day true range volatility for regime detection
    true_range = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    volatility_30d = true_range.rolling(window=30).mean()
    
    # Regime detection using 30-day volatility percentiles
    volatility_percentile = volatility_30d.rolling(window=60).apply(lambda x: (x.iloc[-1] > x.quantile(0.7)), raw=False)
    high_vol_regime = volatility_percentile > 0
    
    # Volume trend persistence: 15-day volume trend consistency
    volume_trend_15d = df['volume'].rolling(window=15).apply(lambda x: (x.iloc[-1] > x.mean()).astype(float), raw=False)
    volume_acceleration_15d = df['volume'].rolling(window=15).mean() / df['volume'].shift(15).rolling(window=15).mean()
    
    # Volume confirmation: combined volume trend and acceleration
    volume_confirmation = volume_trend_15d * volume_acceleration_15d
    
    # Regime-aware smoothing: different window sizes based on volatility regime
    base_factor = enhanced_momentum * volume_confirmation / (volatility_30d + 1e-7)
    
    # Apply regime-specific smoothing
    smoothed_factor = base_factor.copy()
    smoothed_factor[high_vol_regime] = base_factor[high_vol_regime].rolling(window=5).mean()
    smoothed_factor[~high_vol_regime] = base_factor[~high_vol_regime].rolling(window=10).mean()
    
    # Stable multiplicative interaction with regime-adjusted weights
    regime_weight = 1.2 * high_vol_regime.astype(float) + 0.8 * (~high_vol_regime).astype(float)
    alpha_factor = smoothed_factor * volume_confirmation * regime_weight / (volatility_30d + 1e-7)
    
    return alpha_factor
