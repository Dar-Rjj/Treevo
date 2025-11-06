import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility Regime Adaptive Momentum factor
    Combines multi-scale volatility regime detection with regime-adaptive momentum signals
    """
    # Calculate returns
    returns = df['close'].pct_change()
    
    # Multi-scale volatility calculations
    vol_3d = returns.rolling(window=3, min_periods=2).std()
    vol_10d = returns.rolling(window=10, min_periods=5).std()
    
    # Volatility clustering and persistence measures
    abs_returns = returns.abs()
    vol_clustering_3d = abs_returns.rolling(window=3, min_periods=2).apply(
        lambda x: x.autocorr() if len(x) > 1 and not np.isnan(x.autocorr()) else 0
    )
    
    vol_clustering_10d = abs_returns.rolling(window=10, min_periods=5).apply(
        lambda x: x.autocorr() if len(x) > 5 and not np.isnan(x.autocorr()) else 0
    )
    
    # Volatility regime classification
    vol_ratio = vol_3d / vol_10d
    regime_stability = vol_ratio.rolling(window=5, min_periods=3).std()
    
    # High volatility regime indicator (1 for high vol, 0 for low vol)
    high_vol_regime = (vol_3d > vol_3d.rolling(window=20, min_periods=10).quantile(0.7)).astype(int)
    
    # Regime transition detection
    regime_changes = high_vol_regime.diff().abs()
    regime_duration = high_vol_regime.groupby((high_vol_regime != high_vol_regime.shift()).cumsum()).cumcount() + 1
    
    # Multi-scale momentum calculations
    momentum_3d = df['close'] / df['close'].shift(3) - 1
    momentum_10d = df['close'] / df['close'].shift(10) - 1
    momentum_20d = df['close'] / df['close'].shift(20) - 1
    
    # Regime-adaptive momentum measures
    # High volatility regime: focus on momentum persistence
    high_vol_momentum_persistence = momentum_3d.rolling(window=5, min_periods=3).apply(
        lambda x: x.autocorr() if len(x) > 2 and not np.isnan(x.autocorr()) else 0
    )
    
    # Low volatility regime: focus on momentum acceleration
    low_vol_momentum_acceleration = momentum_3d.diff(2).rolling(window=5, min_periods=3).mean()
    
    # Volume analysis
    volume_ma_5 = df['volume'].rolling(window=5, min_periods=3).mean()
    volume_ma_20 = df['volume'].rolling(window=20, min_periods=10).mean()
    volume_ratio = volume_ma_5 / volume_ma_20
    
    # Volume-momentum alignment
    volume_momentum_alignment = np.sign(momentum_3d) * np.sign(volume_ratio - 1)
    
    # Regime-dependent volume characteristics
    high_vol_volume_persistence = df['volume'].pct_change().rolling(window=3, min_periods=2).std()
    low_vol_volume_persistence = df['volume'].rolling(window=5, min_periods=3).std() / df['volume'].rolling(window=20, min_periods=10).mean()
    
    # Volume-regime momentum score
    volume_confirmation_strength = volume_momentum_alignment * volume_ratio
    
    # Regime-weighted momentum calculations
    regime_weight = 1 - regime_stability / (regime_stability.rolling(window=20, min_periods=10).max() + 1e-8)
    
    # High volatility regime momentum (emphasizing persistence)
    high_vol_momentum = momentum_3d * high_vol_momentum_persistence * (1 + volume_confirmation_strength)
    
    # Low volatility regime momentum (emphasizing acceleration)
    low_vol_momentum = momentum_10d * (1 + low_vol_momentum_acceleration) * (1 + volume_confirmation_strength)
    
    # Combine regime-adaptive momentum
    regime_adaptive_momentum = (
        high_vol_regime * high_vol_momentum + 
        (1 - high_vol_regime) * low_vol_momentum
    )
    
    # Apply regime stability weighting
    stable_regime_momentum = regime_adaptive_momentum * regime_weight
    
    # Dynamic momentum thresholds based on volatility
    vol_adjusted_threshold = vol_3d / vol_3d.rolling(window=20, min_periods=10).mean()
    sensitivity_adjustment = 1 / (1 + vol_adjusted_threshold)
    
    # Final regime-aware momentum signal
    final_signal = stable_regime_momentum * sensitivity_adjustment
    
    # Smooth the signal
    factor = final_signal.rolling(window=3, min_periods=2).mean()
    
    return factor
