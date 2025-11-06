import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize result series
    result = pd.Series(index=data.index, dtype=float)
    
    # Calculate rolling windows for volatility calculations
    for i in range(len(data)):
        if i < 8:  # Need at least 8 days for calculations
            result.iloc[i] = 0
            continue
            
        # Bidirectional Volatility Microstructure
        # Directional Volatility Components
        high_open_diff = (data['high'].iloc[i-4:i+1] - data['open'].iloc[i-4:i+1]).values
        open_low_diff = (data['open'].iloc[i-4:i+1] - data['low'].iloc[i-4:i+1]).values
        
        upward_volatility = np.std(high_open_diff) if len(high_open_diff) > 1 else 0.001
        downward_volatility = np.std(open_low_diff) if len(open_low_diff) > 1 else 0.001
        volatility_direction_ratio = upward_volatility / (downward_volatility + 0.001)
        
        # Intraday Volatility Fracture
        morning_vol = data['high'].iloc[i] - data['open'].iloc[i]
        afternoon_vol = data['close'].iloc[i] - data['low'].iloc[i]
        
        # Historical volatility denominators
        hist_high_open = (data['high'].iloc[i-4:i] - data['open'].iloc[i-4:i]).values
        hist_open_low = (data['open'].iloc[i-4:i] - data['low'].iloc[i-4:i]).values
        
        morning_vol_intensity = morning_vol / (np.std(hist_high_open) + 0.001) if len(hist_high_open) > 1 else 0
        afternoon_vol_intensity = afternoon_vol / (np.std(hist_open_low) + 0.001) if len(hist_open_low) > 1 else 0
        volatility_fracture_asymmetry = morning_vol_intensity - afternoon_vol_intensity
        
        # Bidirectional Volume Flow Dynamics
        # Directional Volume Pressure
        high_low_range = data['high'].iloc[i] - data['low'].iloc[i] + 0.001
        upward_volume_pressure = data['amount'].iloc[i] * (data['high'].iloc[i] - data['open'].iloc[i]) / high_low_range
        downward_volume_pressure = data['amount'].iloc[i] * (data['open'].iloc[i] - data['low'].iloc[i]) / high_low_range
        volume_pressure_imbalance = upward_volume_pressure - downward_volume_pressure
        
        # Flow Acceleration Patterns
        if i >= 2:
            vol_ratio_t = data['amount'].iloc[i] / (data['amount'].iloc[i-1] + 0.001) - 1
            vol_ratio_t1 = data['amount'].iloc[i-1] / (data['amount'].iloc[i-2] + 0.001) - 1
            volume_flow_acceleration = vol_ratio_t - vol_ratio_t1
            
            price_flow_t = (data['close'].iloc[i] - data['open'].iloc[i]) / (data['high'].iloc[i] - data['low'].iloc[i] + 0.001)
            price_flow_t1 = (data['close'].iloc[i-1] - data['open'].iloc[i-1]) / (data['high'].iloc[i-1] - data['low'].iloc[i-1] + 0.001)
            price_flow_acceleration = price_flow_t - price_flow_t1
            bidirectional_flow_momentum = volume_flow_acceleration * price_flow_acceleration
        else:
            bidirectional_flow_momentum = 0
        
        # Multi-Timeframe Volatility Regimes
        # Volatility Scale Analysis
        micro_close = data['close'].iloc[i-2:i+1].values
        macro_close = data['close'].iloc[i-8:i+1].values
        
        micro_volatility = np.std(micro_close) if len(micro_close) > 1 else 0.001
        macro_volatility = np.std(macro_close) if len(macro_close) > 1 else 0.001
        volatility_scale_ratio = micro_volatility / (macro_volatility + 0.001)
        
        # Regime Transition Detection
        if i >= 1:
            prev_vol_scale_ratio = np.std(data['close'].iloc[i-3:i].values) / (np.std(data['close'].iloc[i-9:i].values) + 0.001) if i >= 9 else 1
            volatility_expansion = volatility_scale_ratio / (prev_vol_scale_ratio + 0.001)
        else:
            volatility_expansion = 1
        
        # Direction Consistency (count consecutive days with same sign of close-open)
        direction_consistency = 0
        for j in range(min(5, i+1)):
            if i-j < 0:
                break
            current_sign = np.sign(data['close'].iloc[i-j] - data['open'].iloc[i-j])
            if j == 0:
                ref_sign = current_sign
                direction_consistency = 1
            elif current_sign == ref_sign:
                direction_consistency += 1
            else:
                break
        
        regime_transition_signal = volatility_expansion * (1 + direction_consistency / 5)
        
        # Fracture-Volatility Alignment
        volatility_flow_alignment = np.sign(volatility_fracture_asymmetry) * np.sign(volume_pressure_imbalance)
        fracture_intensity = volatility_fracture_asymmetry * volume_pressure_imbalance
        
        # Multi-Scale Fracture Integration
        micro_fracture_component = volatility_flow_alignment * fracture_intensity
        macro_fracture_component = micro_fracture_component * volatility_scale_ratio
        integrated_fracture_signal = macro_fracture_component * regime_transition_signal
        
        # Bidirectional Momentum Structure
        # Directional Momentum Components
        upward_momentum = (data['high'].iloc[i] - data['open'].iloc[i]) / (data['open'].iloc[i] + 0.001)
        downward_momentum = (data['open'].iloc[i] - data['low'].iloc[i]) / (data['open'].iloc[i] + 0.001)
        momentum_direction_bias = upward_momentum - downward_momentum
        
        # Volume-Weighted Momentum
        volume_weighted_upward = upward_momentum * np.log(data['amount'].iloc[i] + 1)
        volume_weighted_downward = downward_momentum * np.log(data['amount'].iloc[i] + 1)
        bidirectional_momentum = volume_weighted_upward - volume_weighted_downward
        
        # Flow-Pressure Dynamics
        # Intraday Pressure Analysis
        opening_pressure = (data['open'].iloc[i] - data['low'].iloc[i]) / high_low_range
        closing_pressure = (data['high'].iloc[i] - data['close'].iloc[i]) / high_low_range
        daily_pressure_shift = opening_pressure - closing_pressure
        
        # Volume Flow Persistence
        volume_direction_persistence = 0
        for j in range(min(5, i+1)):
            if i-j < 1:
                break
            current_sign = np.sign(data['amount'].iloc[i-j] - data['amount'].iloc[i-j-1])
            if j == 0:
                ref_sign = current_sign
                volume_direction_persistence = 1
            elif current_sign == ref_sign:
                volume_direction_persistence += 1
            else:
                break
        
        flow_persistence_factor = 1 + volume_direction_persistence / 5
        persistent_flow_signal = flow_persistence_factor * bidirectional_flow_momentum
        
        # Regime-Adaptive Fracture Behavior
        abs_momentum_bias = abs(momentum_direction_bias)
        
        if volatility_expansion > 1.1 and abs_momentum_bias > 0.015:
            # High Volatility Expansion & Strong Direction Bias
            fracture_amplification = volatility_fracture_asymmetry * volume_pressure_imbalance
            direction_amplitude = abs_momentum_bias * volatility_direction_ratio
            regime_signal = fracture_amplification * direction_amplitude * volatility_flow_alignment
        elif volatility_expansion > 1.1 and abs_momentum_bias <= 0.015:
            # High Volatility Expansion & Weak Direction Bias
            flow_convergence = volume_flow_acceleration * price_flow_acceleration
            pressure_divergence = abs(daily_pressure_shift) * volatility_fracture_asymmetry
            regime_signal = flow_convergence * pressure_divergence * fracture_intensity
        elif volatility_expansion <= 1.1 and abs_momentum_bias > 0.015:
            # Low Volatility Expansion & Strong Direction Bias
            persistent_flow_amplitude = persistent_flow_signal * momentum_direction_bias
            volatility_concentration = volatility_scale_ratio * volatility_direction_ratio
            regime_signal = persistent_flow_amplitude * volatility_concentration * volatility_flow_alignment
        else:
            # Low Volatility Expansion & Weak Direction Bias
            flow_pressure_convergence = persistent_flow_signal * daily_pressure_shift
            volatility_flow_integration = volatility_fracture_asymmetry * volume_flow_acceleration
            regime_signal = flow_pressure_convergence * volatility_flow_integration * fracture_intensity
        
        # Composite Factor Construction
        # Core Volatility Fracture
        fracture_base = integrated_fracture_signal * bidirectional_momentum
        flow_enhanced_fracture = fracture_base * persistent_flow_signal
        regime_adjusted_fracture = flow_enhanced_fracture * regime_signal
        
        # Bidirectional Integration
        direction_adjusted = regime_adjusted_fracture * momentum_direction_bias
        pressure_enhanced = direction_adjusted * daily_pressure_shift
        
        # Final Factor Generation
        volatility_scaled_factor = pressure_enhanced * volatility_scale_ratio
        multi_scale_volatility_fracture_alpha = volatility_scaled_factor * volatility_direction_ratio
        
        result.iloc[i] = multi_scale_volatility_fracture_alpha
    
    return result
