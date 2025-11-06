import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Ensure data is sorted by date
    data = data.sort_index()
    
    # Initialize result series
    result = pd.Series(index=data.index, dtype=float)
    
    for i in range(2, len(data)):
        # Current day data
        open_t = data['open'].iloc[i]
        high_t = data['high'].iloc[i]
        low_t = data['low'].iloc[i]
        close_t = data['close'].iloc[i]
        volume_t = data['volume'].iloc[i]
        amount_t = data['amount'].iloc[i]
        
        # Previous day data
        close_t1 = data['close'].iloc[i-1]
        volume_t1 = data['volume'].iloc[i-1] if i-1 >= 0 else volume_t
        amount_t1 = data['amount'].iloc[i-1] if i-1 >= 0 else amount_t
        
        # Two days ago data
        close_t2 = data['close'].iloc[i-2] if i-2 >= 0 else close_t1
        
        # Avoid division by zero
        high_low_range = high_t - low_t
        if high_low_range == 0:
            high_low_range = 1e-10
        
        # Directional Momentum Fracture
        fracture_intensity = ((close_t - close_t1) / high_low_range) * abs(close_t - close_t2)
        
        # Fracture Persistence (count consecutive days where Fracture Intensity > 0.5)
        fracture_persistence = 0
        for j in range(i, max(-1, i-10), -1):
            if j < 2:
                break
            close_j = data['close'].iloc[j]
            close_j1 = data['close'].iloc[j-1]
            close_j2 = data['close'].iloc[j-2] if j-2 >= 0 else close_j1
            high_j = data['high'].iloc[j]
            low_j = data['low'].iloc[j]
            high_low_j = high_j - low_j
            if high_low_j == 0:
                high_low_j = 1e-10
            fi_j = ((close_j - close_j1) / high_low_j) * abs(close_j - close_j2)
            if fi_j > 0.5:
                fracture_persistence += 1
            else:
                break
        
        # Fracture Acceleration
        if i >= 3:
            close_t3 = data['close'].iloc[i-3]
            high_t1 = data['high'].iloc[i-1]
            low_t1 = data['low'].iloc[i-1]
            high_low_t1 = high_t1 - low_t1
            if high_low_t1 == 0:
                high_low_t1 = 1e-10
            fracture_intensity_t1 = ((close_t1 - close_t2) / high_low_t1) * abs(close_t1 - close_t3)
            fracture_acceleration = (fracture_intensity - fracture_intensity_t1) * np.sign(close_t - close_t1)
        else:
            fracture_acceleration = 0
        
        # Asymmetric Momentum Response
        if close_t > close_t1:
            upward_fracture_strength = (close_t - low_t) / high_low_range
            downward_fracture_resistance = 1.0
        elif close_t < close_t1:
            upward_fracture_strength = 1.0
            downward_fracture_resistance = (high_t - close_t) / high_low_range
        else:
            upward_fracture_strength = 0.5
            downward_fracture_resistance = 0.5
        
        if downward_fracture_resistance == 0:
            downward_fracture_resistance = 1e-10
        fracture_asymmetry_ratio = upward_fracture_strength / downward_fracture_resistance
        
        # Momentum Fracture Regimes
        if fracture_intensity > 0.7 and fracture_persistence > 2:
            fracture_regime = 'strong'
        elif 0.3 <= fracture_intensity <= 0.7:
            fracture_regime = 'moderate'
        else:
            fracture_regime = 'weak'
        
        # Temporal Asymmetry Patterns
        # Opening Asymmetry
        if volume_t1 == 0:
            volume_t1 = 1e-10
        opening_asymmetry = ((open_t - close_t1) / high_low_range) * (volume_t / volume_t1)
        
        # Closing Asymmetry
        if amount_t1 == 0:
            amount_t1 = 1e-10
        closing_asymmetry = ((close_t - open_t) / high_low_range) * (amount_t / amount_t1)
        
        # Midday Asymmetry
        if close_t != open_t:
            midday_asymmetry = ((high_t + low_t) / 2 - open_t) / (close_t - open_t)
        else:
            midday_asymmetry = 0
        
        # Multi-period Asymmetry
        if i >= 3:
            close_diff_1 = close_t - close_t1
            close_diff_2 = close_t1 - close_t2
            if close_diff_2 != 0:
                forward_asymmetry = close_diff_1 / close_diff_2
            else:
                forward_asymmetry = 0
            if close_diff_1 != 0:
                backward_asymmetry = close_diff_2 / close_diff_1
            else:
                backward_asymmetry = 0
            if backward_asymmetry != 0:
                temporal_asymmetry_ratio = forward_asymmetry / backward_asymmetry
            else:
                temporal_asymmetry_ratio = 0
        else:
            temporal_asymmetry_ratio = 1.0
        
        # Asymmetry Persistence
        opening_asymmetry_persistence = 0
        for j in range(i, max(-1, i-10), -1):
            if j < 1:
                break
            open_j = data['open'].iloc[j]
            close_j1 = data['close'].iloc[j-1]
            high_j = data['high'].iloc[j]
            low_j = data['low'].iloc[j]
            volume_j = data['volume'].iloc[j]
            volume_j1 = data['volume'].iloc[j-1] if j-1 >= 0 else volume_j
            high_low_j = high_j - low_j
            if high_low_j == 0:
                high_low_j = 1e-10
            if volume_j1 == 0:
                volume_j1 = 1e-10
            oa_j = ((open_j - close_j1) / high_low_j) * (volume_j / volume_j1)
            if oa_j > 0:
                opening_asymmetry_persistence += 1
            else:
                break
        
        closing_asymmetry_persistence = 0
        for j in range(i, max(-1, i-10), -1):
            if j < 1:
                break
            close_j = data['close'].iloc[j]
            open_j = data['open'].iloc[j]
            high_j = data['high'].iloc[j]
            low_j = data['low'].iloc[j]
            amount_j = data['amount'].iloc[j]
            amount_j1 = data['amount'].iloc[j-1] if j-1 >= 0 else amount_j
            high_low_j = high_j - low_j
            if high_low_j == 0:
                high_low_j = 1e-10
            if amount_j1 == 0:
                amount_j1 = 1e-10
            ca_j = ((close_j - open_j) / high_low_j) * (amount_j / amount_j1)
            if ca_j > 0:
                closing_asymmetry_persistence += 1
            else:
                break
        
        combined_asymmetry_strength = opening_asymmetry_persistence * closing_asymmetry_persistence
        
        # Volume-Momentum Fracture Alignment
        volume_fracture_intensity = (volume_t / volume_t1) * (abs(close_t - close_t1) / high_low_range)
        
        # Volume Fracture Asymmetry (using rolling window)
        volume_up = 0
        volume_down = 0
        count_up = 0
        count_down = 0
        for j in range(max(0, i-20), i+1):
            if j < 1:
                continue
            close_j = data['close'].iloc[j]
            close_j1 = data['close'].iloc[j-1]
            volume_j = data['volume'].iloc[j]
            if close_j > close_j1:
                volume_up += volume_j
                count_up += 1
            elif close_j < close_j1:
                volume_down += volume_j
                count_down += 1
        
        if count_down > 0:
            avg_volume_down = volume_down / count_down
        else:
            avg_volume_down = 1e-10
        
        if count_up > 0:
            avg_volume_up = volume_up / count_up
        else:
            avg_volume_up = 1e-10
        
        volume_fracture_asymmetry = avg_volume_up / avg_volume_down
        
        # Volume Fracture Acceleration
        if i >= 3:
            volume_t2 = data['volume'].iloc[i-2] if i-2 >= 0 else volume_t1
            high_t1 = data['high'].iloc[i-1]
            low_t1 = data['low'].iloc[i-1]
            high_low_t1 = high_t1 - low_t1
            if high_low_t1 == 0:
                high_low_t1 = 1e-10
            vfi_t1 = (volume_t1 / volume_t2) * (abs(close_t1 - close_t2) / high_low_t1)
            volume_fracture_acceleration = (volume_fracture_intensity - vfi_t1) * np.sign(close_t - close_t1)
        else:
            volume_fracture_acceleration = 0
        
        # Amount-Momentum Coupling
        amount_fracture_efficiency = (amount_t / amount_t1) * ((close_t - close_t1) / high_low_range)
        
        if volume_fracture_intensity != 0:
            amount_volume_fracture_ratio = amount_fracture_efficiency / volume_fracture_intensity
        else:
            amount_volume_fracture_ratio = 0
        
        if volume_t != 0:
            fracture_cost_efficiency = (close_t - close_t1) * amount_t / volume_t
        else:
            fracture_cost_efficiency = 0
        
        # Fracture Volume Regimes
        if volume_fracture_intensity > 1.2 and volume_fracture_asymmetry > 1.5:
            volume_fracture_regime = 'high'
        elif 0.8 <= volume_fracture_intensity <= 1.2:
            volume_fracture_regime = 'moderate'
        else:
            volume_fracture_regime = 'low'
        
        # Temporal Fracture Convergence
        ultra_short_fracture = fracture_intensity * opening_asymmetry
        short_term_fracture = volume_fracture_intensity * closing_asymmetry
        medium_term_fracture = temporal_asymmetry_ratio * amount_fracture_efficiency
        
        # Fracture Duration
        fracture_duration = 0
        for j in range(i, max(-1, i-10), -1):
            if j < 2:
                break
            close_j = data['close'].iloc[j]
            close_j1 = data['close'].iloc[j-1]
            close_j2 = data['close'].iloc[j-2] if j-2 >= 0 else close_j1
            high_j = data['high'].iloc[j]
            low_j = data['low'].iloc[j]
            high_low_j = high_j - low_j
            if high_low_j == 0:
                high_low_j = 1e-10
            fi_j = ((close_j - close_j1) / high_low_j) * abs(close_j - close_j2)
            if fi_j > 0.3:
                fracture_duration += 1
            else:
                break
        
        # Volume Fracture Duration
        volume_fracture_duration = 0
        for j in range(i, max(-1, i-10), -1):
            if j < 1:
                break
            volume_j = data['volume'].iloc[j]
            volume_j1 = data['volume'].iloc[j-1] if j-1 >= 0 else volume_j
            close_j = data['close'].iloc[j]
            close_j1 = data['close'].iloc[j-1]
            high_j = data['high'].iloc[j]
            low_j = data['low'].iloc[j]
            high_low_j = high_j - low_j
            if high_low_j == 0:
                high_low_j = 1e-10
            vfi_j = (volume_j / volume_j1) * (abs(close_j - close_j1) / high_low_j)
            if vfi_j > 1.0:
                volume_fracture_duration += 1
            else:
                break
        
        # Fracture Convergence Strength
        fracture_alignments = [ultra_short_fracture, short_term_fracture, medium_term_fracture]
        positive_alignments = sum(1 for fa in fracture_alignments if fa > 0)
        
        if positive_alignments == 3 and fracture_duration > 3:
            convergence_strength = 'strong'
        elif positive_alignments == 2 and fracture_duration > 1:
            convergence_strength = 'moderate'
        else:
            convergence_strength = 'weak'
        
        # Asymmetry-Adaptive Fracture Synthesis
        # Core Fracture Momentum Base
        fracture_momentum_signal = fracture_asymmetry_ratio * volume_fracture_intensity * temporal_asymmetry_ratio
        volume_asymmetry_enhancement = fracture_momentum_signal * amount_fracture_efficiency * volume_fracture_asymmetry
        temporal_confirmation = volume_asymmetry_enhancement * opening_asymmetry * closing_asymmetry
        
        # Fracture Regime Weighting
        if fracture_regime == 'strong':
            regime_weight = 1.3
        elif fracture_regime == 'moderate':
            regime_weight = 1.0
        else:
            regime_weight = 0.7
        
        core_signal = temporal_confirmation * regime_weight
        
        # Asymmetry-Momentum Validation
        if fracture_acceleration > 0.1:
            asymmetry_boost = 0.4 * fracture_acceleration
        elif fracture_acceleration > 0:
            asymmetry_boost = 0.2 * fracture_acceleration
        else:
            asymmetry_boost = 0
        
        asymmetry_breakdown_penalty = -0.5 * abs(temporal_asymmetry_ratio - 1)
        
        # Convergence Multiplier
        if convergence_strength == 'strong':
            convergence_multiplier = 1.6
        elif convergence_strength == 'moderate':
            convergence_multiplier = 1.1
        else:
            convergence_multiplier = 0.7
        
        # Final Alpha Calculation
        final_alpha = (core_signal + asymmetry_boost + asymmetry_breakdown_penalty) * convergence_multiplier
        
        # Apply bounds and normalization
        if final_alpha > 1.0:
            final_alpha = 1.0
        elif final_alpha < -1.0:
            final_alpha = -1.0
        
        result.iloc[i] = final_alpha
    
    # Fill NaN values with 0
    result = result.fillna(0)
    
    return result
