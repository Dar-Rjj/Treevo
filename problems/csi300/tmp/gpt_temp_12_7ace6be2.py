import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    for i in range(10, len(df)):
        current_data = df.iloc[:i+1].copy()
        
        # Fractal Volatility Structure
        # Intraday Volatility Asymmetry
        intraday_vol_asymmetry = (current_data['high'].iloc[-1] - current_data['open'].iloc[-1]) / \
                                (current_data['open'].iloc[-1] - current_data['low'].iloc[-1] + 1e-8)
        
        # Fractal Volatility Persistence
        if i >= 2:
            vol_persistence = ((current_data['high'].iloc[-1] - current_data['low'].iloc[-1]) * \
                             (current_data['high'].iloc[-2] - current_data['low'].iloc[-2])) / \
                            (current_data['high'].iloc[-3] - current_data['low'].iloc[-3] + 1e-8)
        else:
            vol_persistence = 1.0
        
        # Volatility Compression
        if i >= 1:
            current_range = current_data['high'].iloc[-1] - current_data['low'].iloc[-1]
            prev_range = current_data['high'].iloc[-2] - current_data['low'].iloc[-2]
            vol_compression = min(current_range, prev_range) / (max(current_range, prev_range) + 1e-8)
        else:
            vol_compression = 1.0
        
        # Fractal Volatility Signature
        fractal_vol_signature = intraday_vol_asymmetry * vol_persistence * vol_compression
        
        # Multi-Horizon Momentum Asymmetry
        # Momentum Horizon Structure
        if i >= 2:
            ultra_short_momentum = (current_data['close'].iloc[-1] / current_data['close'].iloc[-2]) - \
                                  (current_data['close'].iloc[-2] / current_data['close'].iloc[-3])
        else:
            ultra_short_momentum = 0.0
            
        if i >= 4:
            short_term_momentum = (current_data['close'].iloc[-1] / current_data['close'].iloc[-3]) - \
                                 (current_data['close'].iloc[-3] / current_data['close'].iloc[-5])
        else:
            short_term_momentum = 0.0
            
        if i >= 10:
            medium_term_momentum = (current_data['close'].iloc[-1] / current_data['close'].iloc[-6]) - \
                                  (current_data['close'].iloc[-6] / current_data['close'].iloc[-11])
        else:
            medium_term_momentum = 0.0
            
        momentum_horizon_ratio = ultra_short_momentum * short_term_momentum / (medium_term_momentum + 1e-8)
        
        # Momentum Asymmetry Framework
        bull_momentum_intensity = 0.0
        bear_momentum_intensity = 0.0
        
        for j in range(max(0, i-4), i+1):
            if current_data['close'].iloc[j] > current_data['open'].iloc[j]:
                bull_momentum_intensity += (current_data['close'].iloc[j] - current_data['open'].iloc[j]) * current_data['volume'].iloc[j]
            elif current_data['close'].iloc[j] < current_data['open'].iloc[j]:
                bear_momentum_intensity += (current_data['open'].iloc[j] - current_data['close'].iloc[j]) * current_data['volume'].iloc[j]
        
        momentum_asymmetry_ratio = bull_momentum_intensity / (bear_momentum_intensity + 1e-8)
        
        # Momentum Reversal Strength
        if i >= 3:
            reversal_strength = ((current_data['close'].iloc[-1] - current_data['close'].iloc[-4]) / \
                               (current_data['high'].iloc[-4] - current_data['low'].iloc[-4] + 1e-8)) - \
                              ((current_data['close'].iloc[-1] - current_data['close'].iloc[-2]) / \
                               (current_data['high'].iloc[-2] - current_data['low'].iloc[-2] + 1e-8))
        else:
            reversal_strength = 0.0
        
        # Fractal Momentum Integration
        horizon_asymmetry_composite = momentum_horizon_ratio * momentum_asymmetry_ratio
        volatility_momentum_fractal = fractal_vol_signature * horizon_asymmetry_composite
        momentum_transition_signal = np.sign(ultra_short_momentum) * np.sign(short_term_momentum) * np.sign(medium_term_momentum)
        
        # Fractal Volume-Price Dynamics
        # Volume Fractal Structure
        if i >= 2:
            volume_acceleration = (current_data['volume'].iloc[-1] / current_data['volume'].iloc[-2]) - \
                                (current_data['volume'].iloc[-2] / current_data['volume'].iloc[-3])
        else:
            volume_acceleration = 0.0
            
        if i >= 4:
            volume_persistence = (current_data['volume'].iloc[-1] / current_data['volume'].iloc[-3]) - \
                               (current_data['volume'].iloc[-3] / current_data['volume'].iloc[-5])
        else:
            volume_persistence = 0.0
            
        volume_momentum_score = volume_acceleration * volume_persistence
        
        if i >= 2:
            fractal_volume_pattern = current_data['volume'].iloc[-1] * current_data['volume'].iloc[-2] / (current_data['volume'].iloc[-3] + 1e-8)
        else:
            fractal_volume_pattern = 1.0
        
        # Price-Volume Asymmetry
        if i >= 1:
            volume_price_correlation = np.sign(current_data['close'].iloc[-1] - current_data['close'].iloc[-2]) * \
                                     np.sign(current_data['volume'].iloc[-1] - current_data['volume'].iloc[-2])
        else:
            volume_price_correlation = 0.0
            
        volume_efficiency = ((current_data['close'].iloc[-1] - current_data['open'].iloc[-1]) / \
                           (current_data['high'].iloc[-1] - current_data['low'].iloc[-1] + 1e-8)) * current_data['volume'].iloc[-1]
        
        if i >= 1:
            price_volume_divergence = ((current_data['close'].iloc[-1] - current_data['close'].iloc[-2]) / \
                                     (current_data['high'].iloc[-2] - current_data['low'].iloc[-2] + 1e-8)) - \
                                    ((current_data['volume'].iloc[-1] - current_data['volume'].iloc[-2]) / \
                                     (current_data['volume'].iloc[-2] + 1e-8))
        else:
            price_volume_divergence = 0.0
            
        volume_sum_3 = sum(current_data['volume'].iloc[max(0, i-2):i+1])
        volume_concentration = current_data['volume'].iloc[-1] / (volume_sum_3 + 1e-8)
        
        # Fractal Volume-Price Integration
        volume_fractal_momentum = volume_momentum_score * ultra_short_momentum
        fractal_price_volume_alignment = fractal_volume_pattern * volume_price_correlation
        multi_fractal_volume_signal = volume_fractal_momentum * fractal_price_volume_alignment
        
        # Range Breakout with Fractal Dynamics
        # Fractal Range Analysis
        if i >= 3:
            range_expansion_signal = (current_data['high'].iloc[-1] - current_data['low'].iloc[-1]) / \
                                   (current_data['high'].iloc[-4] - current_data['low'].iloc[-4] + 1e-8)
        else:
            range_expansion_signal = 1.0
            
        if i >= 1:
            breakout_direction = np.sign(current_data['close'].iloc[-1] - (current_data['high'].iloc[-2] + current_data['low'].iloc[-2])/2)
            breakout_strength = abs(current_data['close'].iloc[-1] - (current_data['high'].iloc[-2] + current_data['low'].iloc[-2])/2) / \
                              (current_data['high'].iloc[-2] - current_data['low'].iloc[-2] + 1e-8)
        else:
            breakout_direction = 0.0
            breakout_strength = 0.0
            
        fractal_breakout_confirmation = breakout_direction * breakout_strength * range_expansion_signal
        
        # Volatility-Breakout Integration
        volatility_breakout_asymmetry = fractal_breakout_confirmation * vol_compression
        range_volatility_fractal = range_expansion_signal * fractal_vol_signature
        breakout_momentum = fractal_breakout_confirmation * momentum_transition_signal
        
        # Multi-Timeframe Breakout
        short_term_breakout = fractal_breakout_confirmation * ultra_short_momentum
        medium_term_breakout = range_volatility_fractal * short_term_momentum
        long_term_breakout = volatility_breakout_asymmetry * medium_term_momentum
        
        # Regime Classification & Asymmetry Enhancement
        # Volatility Regime Detection
        if vol_persistence > 1.1:
            volatility_regime = 'expanding'
        elif vol_persistence < 0.9:
            volatility_regime = 'contracting'
        else:
            volatility_regime = 'neutral'
        
        # Momentum Regime Classification
        accelerating_momentum = (ultra_short_momentum > 0) and (short_term_momentum > 0) and (medium_term_momentum > 0)
        decelerating_momentum = (ultra_short_momentum < 0) and (short_term_momentum < 0) and (medium_term_momentum < 0)
        strong_momentum = momentum_asymmetry_ratio > 1.5
        
        if accelerating_momentum:
            momentum_regime = 'accelerating'
        elif decelerating_momentum:
            momentum_regime = 'decelerating'
        else:
            momentum_regime = 'mixed'
        
        # Volume Regime Detection
        if volume_concentration > 0.4:
            volume_regime = 'high'
        elif volume_concentration < 0.25:
            volume_regime = 'low'
        else:
            volume_regime = 'normal'
        
        # Asymmetry Enhancement Layer
        volatility_momentum_asymmetry = intraday_vol_asymmetry * reversal_strength
        volume_range_asymmetry = volume_price_correlation * range_expansion_signal
        fractal_asymmetry_multiplier = 1 + abs(volatility_momentum_asymmetry) + abs(volume_range_asymmetry)
        multi_fractal_asymmetry = fractal_asymmetry_multiplier * multi_fractal_volume_signal
        
        # Regime-Adaptive Alpha Construction
        # Expanding Volatility Alpha Components
        volatility_expansion_alpha = volatility_momentum_fractal * fractal_breakout_confirmation
        momentum_acceleration_alpha = horizon_asymmetry_composite * momentum_transition_signal
        expanding_volatility_alpha = volatility_expansion_alpha * momentum_acceleration_alpha * fractal_asymmetry_multiplier
        
        # Contracting Volatility Alpha Components
        volatility_compression_alpha = vol_compression * reversal_strength
        volume_efficiency_alpha = volume_efficiency * multi_fractal_volume_signal
        contracting_volatility_alpha = volatility_compression_alpha * volume_efficiency_alpha * volume_price_correlation
        
        # Neutral Volatility Alpha Components
        range_momentum_alpha = long_term_breakout * volume_fractal_momentum
        fractal_alignment_alpha = fractal_price_volume_alignment * (1 - abs(momentum_transition_signal))
        neutral_volatility_alpha = range_momentum_alpha * fractal_alignment_alpha * price_volume_divergence
        
        # Momentum-Regime Specific Alpha
        accelerating_regime_alpha = expanding_volatility_alpha * momentum_horizon_ratio
        decelerating_regime_alpha = contracting_volatility_alpha * (1 - abs(momentum_transition_signal))
        mixed_regime_alpha = neutral_volatility_alpha * volume_fractal_momentum
        transition_alpha = ((expanding_volatility_alpha + contracting_volatility_alpha) / 2) * volume_price_correlation
        
        # Final Fractal Volatility-Momentum Alpha Synthesis
        # Regime-Based Selection
        if volatility_regime == 'expanding' and momentum_regime == 'accelerating':
            selected_alpha = accelerating_regime_alpha
        elif volatility_regime == 'contracting' and momentum_regime == 'decelerating':
            selected_alpha = decelerating_regime_alpha
        elif volatility_regime == 'neutral' and momentum_regime == 'mixed':
            selected_alpha = mixed_regime_alpha
        elif volatility_regime == 'expanding' and strong_momentum:
            selected_alpha = expanding_volatility_alpha * momentum_asymmetry_ratio
        else:
            selected_alpha = transition_alpha
        
        # Volume Confirmation Layer
        if volume_regime == 'high':
            volume_confirmation = selected_alpha * volume_concentration
        elif volume_regime == 'normal':
            volume_confirmation = selected_alpha * volume_efficiency
        else:  # low volume
            volume_confirmation = selected_alpha * price_volume_divergence
        
        # Fractal Enhancement
        volatility_fractal_enhancement = volume_confirmation * fractal_vol_signature
        momentum_fractal_enhancement = volume_confirmation * horizon_asymmetry_composite
        multi_fractal_enhancement = volatility_fractal_enhancement * momentum_fractal_enhancement
        
        # Final Alpha Output
        asymmetry_weighted_alpha = multi_fractal_enhancement * fractal_asymmetry_multiplier
        regime_adaptive_alpha = asymmetry_weighted_alpha * (1 + abs(momentum_transition_signal))
        final_alpha = regime_adaptive_alpha * multi_fractal_asymmetry
        
        alpha.iloc[i] = final_alpha
    
    # Fill NaN values with 0
    alpha = alpha.fillna(0)
    
    return alpha
