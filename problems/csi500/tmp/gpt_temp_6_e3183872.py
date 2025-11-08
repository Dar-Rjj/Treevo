import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Momentum with Dynamic Regime Adaptation alpha factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize result series
    alpha = pd.Series(index=data.index, dtype=float)
    
    # Calculate required lookback periods
    for i in range(len(data)):
        if i < 29:  # Need at least 30 days for calculations
            alpha.iloc[i] = 0
            continue
            
        current_data = data.iloc[:i+1]
        
        # Multi-Timeframe Momentum Framework
        # Ultra-Short Term (1-2 days)
        momentum_1d = current_data['close'].iloc[-1] - current_data['open'].iloc[-1]
        momentum_2d = (current_data['close'].iloc[-1] - current_data['close'].iloc[-2]) + \
                     (current_data['close'].iloc[-2] - current_data['close'].iloc[-3])
        range_2d = np.mean([current_data['high'].iloc[-1] - current_data['low'].iloc[-1],
                           current_data['high'].iloc[-2] - current_data['low'].iloc[-2]])
        
        # Short Term (3-5 days)
        momentum_3d = sum(current_data['close'].iloc[-3:] - current_data['open'].iloc[-3:])
        momentum_5d = sum(current_data['close'].iloc[-5:] - current_data['open'].iloc[-5:])
        range_5d = np.mean([current_data['high'].iloc[j] - current_data['low'].iloc[j] 
                           for j in range(-5, 0)])
        
        # Medium Term (10-15 days)
        momentum_10d = sum(current_data['close'].iloc[-10:] - current_data['open'].iloc[-10:])
        momentum_15d = sum(current_data['close'].iloc[-15:] - current_data['open'].iloc[-15:])
        range_15d = np.mean([current_data['high'].iloc[j] - current_data['low'].iloc[j] 
                            for j in range(-15, 0)])
        
        # Long Term (20-30 days)
        momentum_20d = sum(current_data['close'].iloc[-20:] - current_data['open'].iloc[-20:])
        momentum_30d = sum(current_data['close'].iloc[-30:] - current_data['open'].iloc[-30:])
        range_30d = np.mean([current_data['high'].iloc[j] - current_data['low'].iloc[j] 
                            for j in range(-30, 0)])
        
        # Volume Analysis Framework
        # Multi-Timeframe Volume
        volume_1d = current_data['volume'].iloc[-1]
        volume_5d = np.mean(current_data['volume'].iloc[-5:])
        volume_15d = np.mean(current_data['volume'].iloc[-15:])
        volume_30d = np.mean(current_data['volume'].iloc[-30:])
        
        # Volume Momentum
        volume_change_1d = volume_1d / current_data['volume'].iloc[-2] - 1 if i > 0 else 0
        volume_change_5d = volume_5d / volume_15d - 1
        volume_change_15d = volume_15d / volume_30d - 1
        
        # Volume-Price Alignment
        daily_alignment = np.sign(momentum_1d) * np.sign(volume_change_1d)
        short_term_alignment = np.sign(momentum_5d) * np.sign(volume_change_5d)
        medium_term_alignment = np.sign(momentum_15d) * np.sign(volume_change_15d)
        
        # Dynamic Regime Detection
        # Volatility Regime System
        v_ratio_5_15 = range_5d / range_15d if range_15d != 0 else 1
        v_ratio_15_30 = range_15d / range_30d if range_30d != 0 else 1
        v_ratio_5_30 = range_5d / range_30d if range_30d != 0 else 1
        
        if v_ratio_5_15 > 1.3:
            volatility_regime = 'high'
            volatility_scale = 0.6
        elif v_ratio_5_15 >= 0.7:
            volatility_regime = 'normal'
            volatility_scale = 1.0
        else:
            volatility_regime = 'low'
            volatility_scale = 1.4
        
        # Volume Regime System
        vol_ratio_5_15 = volume_5d / volume_15d if volume_15d != 0 else 1
        vol_ratio_15_30 = volume_15d / volume_30d if volume_30d != 0 else 1
        vol_ratio_5_30 = volume_5d / volume_30d if volume_30d != 0 else 1
        
        if vol_ratio_5_15 > 1.2:
            volume_regime = 'high'
            volume_scale = 1.3
        elif vol_ratio_5_15 >= 0.8:
            volume_regime = 'normal'
            volume_scale = 1.0
        else:
            volume_regime = 'low'
            volume_scale = 0.7
        
        # Momentum Regime System
        ultra_short_agreement = np.sign(momentum_1d) == np.sign(momentum_2d)
        short_medium_agreement = np.sign(momentum_5d) == np.sign(momentum_15d)
        medium_long_agreement = np.sign(momentum_15d) == np.sign(momentum_30d)
        
        consistency_score = sum([ultra_short_agreement, short_medium_agreement, medium_long_agreement])
        
        if consistency_score == 3:
            momentum_consistency_scale = 1.6
        elif consistency_score == 2:
            momentum_consistency_scale = 1.0
        else:
            momentum_consistency_scale = 0.4
        
        # Momentum Persistence
        direction_persistence_length = 1
        strength_persistence_length = 1
        
        # Calculate direction persistence (same sign streak)
        for j in range(2, min(10, i+1)):
            prev_momentum = current_data['close'].iloc[-j] - current_data['open'].iloc[-j]
            if np.sign(prev_momentum) == np.sign(momentum_1d):
                direction_persistence_length += 1
            else:
                break
        
        # Calculate strength persistence (increasing absolute momentum streak)
        for j in range(2, min(6, i+1)):
            prev_momentum = abs(current_data['close'].iloc[-j] - current_data['open'].iloc[-j])
            current_prev_momentum = abs(current_data['close'].iloc[-(j-1)] - current_data['open'].iloc[-(j-1)])
            if current_prev_momentum > prev_momentum:
                strength_persistence_length += 1
            else:
                break
        
        # Factor Construction Engine
        # Base Momentum Calculation
        weighted_momentum = (4 * momentum_2d + 3 * momentum_5d + 2 * momentum_15d + 1 * momentum_30d) / 10
        
        # Volume-Adjusted Momentum
        volume_weight = volume_1d / volume_15d if volume_15d != 0 else 1
        base_factor = weighted_momentum * volume_weight
        
        # Persistence Enhancement
        persistence_multiplier = 1 + (direction_persistence_length / 8)
        enhanced_base = base_factor * persistence_multiplier
        
        strength_multiplier = 1 + (strength_persistence_length / 5)
        further_enhanced = enhanced_base * strength_multiplier
        
        # Regime-Based Scaling
        regime_scaled = further_enhanced * volatility_scale * volume_scale * momentum_consistency_scale
        
        # Alignment Confirmation
        positive_alignment_count = sum([daily_alignment > 0, short_term_alignment > 0, medium_term_alignment > 0])
        alignment_boost = 1 + (positive_alignment_count * 0.15)
        
        # Final Alpha Factor
        final_factor = regime_scaled * alignment_boost
        
        alpha.iloc[i] = final_factor
    
    return alpha
