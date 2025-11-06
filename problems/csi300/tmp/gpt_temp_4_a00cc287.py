import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price-Volume Divergence Momentum with Regime Switching alpha factor
    """
    data = df.copy()
    
    # Helper function for ATR calculation
    def calculate_atr(high, low, close, window=14):
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()
        return atr
    
    # Calculate ATR for normalization
    atr = calculate_atr(data['high'], data['low'], data['close'])
    
    # 1. Compute Price-Volume Divergence
    # Directional Price Movement
    signed_price_change = data['close'] - data['close'].shift(1)
    directional_magnitude = abs(signed_price_change)
    
    # Volume-Price Alignment
    volume_weighted_movement = signed_price_change * data['volume']
    cum_volume_weighted = volume_weighted_movement.rolling(window=3).sum()
    
    # Divergence Score
    alignment_ratio = np.where(directional_magnitude != 0, 
                              cum_volume_weighted / (directional_magnitude * data['volume'].rolling(window=3).mean()),
                              0)
    
    # Divergence Signal
    divergence_strength = abs(directional_magnitude - alignment_ratio) / (atr + 1e-8)
    divergence_direction = np.where(alignment_ratio > 0, 1, -1) * divergence_strength
    
    # 2. Compute Multi-Timeframe Momentum
    # Short-Term Momentum (3-day)
    price_momentum_3d = data['close'] / data['close'].shift(3) - 1
    volume_momentum_3d = data['volume'] / data['volume'].shift(3) - 1
    short_term_momentum = price_momentum_3d * (1 + volume_momentum_3d)
    
    # Medium-Term Momentum (8-day)
    price_momentum_8d = data['close'] / data['close'].shift(8) - 1
    volume_momentum_8d = data['volume'] / data['volume'].shift(8) - 1
    medium_term_momentum = price_momentum_8d * (1 + volume_momentum_8d)
    
    # Momentum Hierarchy
    momentum_acceleration = short_term_momentum - medium_term_momentum
    momentum_regime = np.where(abs(momentum_acceleration) > momentum_acceleration.rolling(window=20).std(), 
                              np.sign(momentum_acceleration), 0)
    
    # 3. Implement Regime Switching Logic
    # Market Regime Detection
    volatility_5d = (data['high'] - data['low']).rolling(window=5).std()
    volatility_15d = (data['high'] - data['low']).rolling(window=15).std()
    volatility_ratio = volatility_5d / (volatility_15d + 1e-8)
    
    # Trend vs Mean-reversion detection
    price_persistence = data['close'].rolling(window=5).apply(lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) > 1 else 0)
    volume_consistency = data['volume'].rolling(window=5).std() / (data['volume'].rolling(window=5).mean() + 1e-8)
    
    # Regime Classification
    high_vol_trending = (volatility_ratio > 1.2) & (abs(price_persistence) > 0.3) & (volume_consistency < 0.5)
    low_vol_ranging = (volatility_ratio < 0.8) & (abs(price_persistence) < 0.2)
    transition_regime = ~high_vol_trending & ~low_vol_ranging
    
    # 4. Construct Composite Alpha Factor
    # Regime-specific weighting
    momentum_weight = np.where(high_vol_trending, 0.7, 
                              np.where(low_vol_ranging, 0.3, 0.5))
    divergence_weight = 1 - momentum_weight
    
    # Combine components
    raw_alpha = (momentum_weight * (short_term_momentum + 0.3 * momentum_acceleration) + 
                 divergence_weight * divergence_direction)
    
    # Temporal Enhancement
    momentum_persistence = raw_alpha.rolling(window=3).mean()
    mean_reversion_correction = -0.2 * raw_alpha.rolling(window=10).mean()
    regime_persistence = np.where(high_vol_trending.rolling(window=3).sum() >= 2, 1.1,
                                 np.where(low_vol_ranging.rolling(window=3).sum() >= 2, 0.9, 1.0))
    
    enhanced_alpha = (raw_alpha + momentum_persistence + mean_reversion_correction) * regime_persistence
    
    # Final Factor Refinement
    # Asymmetric filtering
    positive_filter = enhanced_alpha.rolling(window=5).apply(lambda x: x[x > 0].mean() if len(x[x > 0]) > 0 else 0)
    negative_filter = enhanced_alpha.rolling(window=5).apply(lambda x: x[x < 0].mean() if len(x[x < 0]) > 0 else 0)
    
    smoothed_alpha = np.where(enhanced_alpha > 0, positive_filter, negative_filter)
    
    # Regime-dependent scaling
    regime_scale = np.where(high_vol_trending, 1.2,
                           np.where(low_vol_ranging, 0.8, 1.0))
    
    final_factor = smoothed_alpha * regime_scale
    
    # Directional bias adjustment based on momentum regime
    directional_bias = np.where(momentum_regime == 1, 1.1,
                               np.where(momentum_regime == -1, 0.9, 1.0))
    
    final_factor = final_factor * directional_bias
    
    return pd.Series(final_factor, index=data.index)
