import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Regime Momentum-Volume Divergence Factor
    Combines price momentum, volume patterns, and regime detection for enhanced return prediction
    """
    
    # Extract price and volume data
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # 1. REGIME DETECTION COMPONENT
    # Price Regime Identification
    # Calculate price acceleration (second derivative)
    price_change_1 = close.pct_change(1)
    price_change_2 = close.pct_change(2)
    price_acceleration = price_change_1 - price_change_2.shift(1)
    
    # Volatility regime classification
    short_vol = close.pct_change().rolling(window=5).std()
    medium_vol = close.pct_change().rolling(window=20).std()
    vol_ratio = short_vol / medium_vol
    volatility_regime = np.where(vol_ratio > 1.2, 2, np.where(vol_ratio < 0.8, 0, 1))  # 0: low, 1: normal, 2: high
    
    # Volume regime identification
    volume_ma_short = volume.rolling(window=5).mean()
    volume_ma_medium = volume.rolling(window=20).mean()
    volume_spike = volume / volume_ma_medium
    volume_trend = volume_ma_short / volume_ma_medium
    
    # Volume-price divergence
    price_momentum_short = close.pct_change(3)
    volume_momentum_short = volume.pct_change(3)
    volume_price_divergence = np.sign(price_momentum_short) != np.sign(volume_momentum_short)
    
    # 2. MULTI-TIMEFRAME MOMENTUM ENGINE
    # Non-linear momentum calculations
    # Short-term momentum (t-3 to t)
    mom_short = close.pct_change(3)
    mom_short_accel = mom_short - close.pct_change(6).shift(3)
    
    # Medium-term momentum (t-8 to t)
    mom_medium = close.pct_change(8)
    mom_medium_curvature = mom_medium - 2 * close.pct_change(4).shift(4) + close.pct_change(8).shift(8)
    
    # Momentum interaction terms
    mom_interaction = mom_short * mom_medium
    mom_accel_ratio = mom_short_accel / (mom_medium_curvature + 1e-8)
    
    # Volatility-scaled momentum
    # Regime-adaptive volatility measures
    def regime_volatility(regime):
        if regime == 2:  # High volatility
            return close.pct_change().rolling(window=10).std()
        else:  # Low/Normal volatility
            return close.pct_change().rolling(window=20).std()
    
    conditional_vol = pd.Series([regime_volatility(regime)[i] for i, regime in enumerate(volatility_regime)], 
                               index=close.index)
    
    # Dynamic volatility adjustment
    vol_scaling = 1 / (conditional_vol + 1e-8)
    regime_scaling = np.where(volatility_regime == 2, 0.7, np.where(volatility_regime == 0, 1.3, 1.0))
    scaled_mom_short = mom_short * vol_scaling * regime_scaling
    scaled_mom_medium = mom_medium * vol_scaling * regime_scaling
    
    # 3. VOLUME CONFIRMATION SYSTEM
    # Multi-timeframe volume patterns
    vol_mom_short = volume.pct_change(2)
    vol_trend_medium = volume.rolling(window=5).mean() / volume.rolling(window=10).mean() - 1
    vol_accel = vol_mom_short - volume.pct_change(4).shift(2)
    
    # Volume-price convergence/divergence
    volume_weighted_mom = (mom_short * volume) / (volume.rolling(window=5).mean() + 1e-8)
    
    # Detect divergences
    price_vol_correlation = close.pct_change(3).rolling(window=10).corr(volume.pct_change(3))
    divergence_signal = np.where(price_vol_correlation < -0.3, -1, 
                                np.where(price_vol_correlation > 0.3, 1, 0))
    
    # 4. COMPOSITE FACTOR CONSTRUCTION
    # Regime-weighted signal combination
    def regime_weights(regime):
        if regime == 2:  # High volatility - emphasize short-term signals
            return [0.6, 0.3, 0.1]  # [short_mom, volume_conf, regime_signal]
        elif regime == 0:  # Low volatility - emphasize medium-term signals
            return [0.3, 0.5, 0.2]
        else:  # Normal volatility - balanced approach
            return [0.4, 0.4, 0.2]
    
    # Calculate regime-specific weights
    weights = pd.DataFrame([regime_weights(regime) for regime in volatility_regime], 
                          index=close.index, columns=['mom_weight', 'vol_weight', 'regime_weight'])
    
    # Non-linear signal enhancement
    def sigmoid_transform(x):
        return 2 / (1 + np.exp(-2 * x)) - 1
    
    enhanced_mom_short = sigmoid_transform(scaled_mom_short)
    enhanced_mom_medium = sigmoid_transform(scaled_mom_medium)
    
    # Volume confirmation strength adjustment
    vol_confirmation_strength = np.where(volume_spike > 1.5, 1.5, 
                                        np.where(volume_spike < 0.7, 0.7, 1.0))
    
    # Composite momentum signal
    composite_momentum = (enhanced_mom_short * 0.6 + enhanced_mom_medium * 0.4 + 
                         mom_interaction * 0.1 + mom_accel_ratio * 0.1)
    
    # Volume confirmation signal
    volume_signal = (vol_mom_short * 0.4 + vol_trend_medium * 0.3 + 
                    volume_weighted_mom * 0.2 + divergence_signal * 0.1) * vol_confirmation_strength
    
    # Regime context signal
    regime_signal = (price_acceleration * 0.3 + np.where(volume_price_divergence, -0.2, 0.1) + 
                    np.where(volatility_regime == 2, -0.1, np.where(volatility_regime == 0, 0.1, 0)))
    
    # Final factor construction with regime weighting
    factor = (composite_momentum * weights['mom_weight'] + 
             volume_signal * weights['vol_weight'] + 
             regime_signal * weights['regime_weight'])
    
    # Apply final non-linear transformation
    final_factor = sigmoid_transform(factor)
    
    return final_factor
