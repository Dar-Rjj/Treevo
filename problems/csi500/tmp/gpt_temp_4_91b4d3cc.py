import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    for i in range(8, len(df)):
        current_data = df.iloc[:i+1].copy()
        
        # Bidirectional Pressure Asymmetry
        high_t = current_data['high'].iloc[-1]
        low_t = current_data['low'].iloc[-1]
        open_t = current_data['open'].iloc[-1]
        close_t = current_data['close'].iloc[-1]
        
        upward_pressure = (high_t - open_t) / (high_t - low_t) if (high_t - low_t) > 0 else 0
        downward_pressure = (open_t - low_t) / (high_t - low_t) if (high_t - low_t) > 0 else 0
        pressure_asymmetry = (upward_pressure - downward_pressure) / (upward_pressure + downward_pressure + 1e-8)
        
        # Microstructure Fractal Dimensions
        high_t_1 = current_data['high'].iloc[-2]
        low_t_1 = current_data['low'].iloc[-2]
        close_t_1 = current_data['close'].iloc[-2]
        
        intraday_fractal = (high_t - low_t) / (high_t_1 - low_t_1) if (high_t_1 - low_t_1) > 0 else 1
        opening_fractal_gap = abs(open_t - close_t_1) / (high_t_1 - low_t_1) if (high_t_1 - low_t_1) > 0 else 0
        closing_fractal_momentum = (close_t - (high_t + low_t)/2) / (high_t - low_t) if (high_t - low_t) > 0 else 0
        
        # Volume Microstructure Patterns (simplified using daily volume)
        volume_t = current_data['volume'].iloc[-1]
        volume_t_1 = current_data['volume'].iloc[-2]
        volume_t_2 = current_data['volume'].iloc[-3]
        
        volume_fractal_persistence = np.sign(volume_t - volume_t_1) * np.sign(volume_t_1 - volume_t_2)
        
        # Directional Momentum Asymmetry
        upward_momentum = (close_t - low_t) / (high_t - low_t) if (high_t - low_t) > 0 else 0
        downward_momentum = (high_t - close_t) / (high_t - low_t) if (high_t - low_t) > 0 else 0
        momentum_asymmetry = upward_momentum - downward_momentum
        
        # Multi-Timeframe Asymmetry
        close_t_3 = current_data['close'].iloc[-4]
        close_t_8 = current_data['close'].iloc[-9]
        
        # Rolling windows for high/low ranges
        high_3d = current_data['high'].iloc[-4:].max()
        low_3d = current_data['low'].iloc[-4:].min()
        high_8d = current_data['high'].iloc[-9:].max()
        low_8d = current_data['low'].iloc[-9:].min()
        
        ultra_short_asymmetry = (close_t - close_t_1) / (high_t - low_t) if (high_t - low_t) > 0 else 0
        short_term_asymmetry = (close_t - close_t_3) / (high_3d - low_3d) if (high_3d - low_3d) > 0 else 0
        medium_term_asymmetry = (close_t - close_t_8) / (high_8d - low_8d) if (high_8d - low_8d) > 0 else 0
        
        # Asymmetry Convergence Dynamics
        asymmetry_direction_consistency = np.sign(momentum_asymmetry) * np.sign(pressure_asymmetry)
        asymmetry_magnitude_convergence = abs(momentum_asymmetry) * abs(pressure_asymmetry)
        multi_timeframe_alignment = np.sign(ultra_short_asymmetry) + np.sign(short_term_asymmetry) + np.sign(medium_term_asymmetry)
        
        # Fractal Regime Detection Framework
        fractal_compression_ratio = intraday_fractal / (opening_fractal_gap + 1e-8)
        volume_regime_signature = volume_fractal_persistence * 0.5  # Simplified late session momentum
        price_regime_fractal = closing_fractal_momentum * intraday_fractal
        
        fractal_expansion_signal = intraday_fractal * opening_fractal_gap
        volume_regime_shift = 0.5 * volume_fractal_persistence  # Simplified
        price_regime_transition = closing_fractal_momentum * pressure_asymmetry
        
        high_fractal_regime = fractal_compression_ratio * fractal_expansion_signal
        volume_driven_regime = volume_regime_signature * volume_regime_shift
        price_led_regime = price_regime_fractal * price_regime_transition
        
        # Asymmetric Volume-Price Integration
        volume_enhanced_asymmetry = momentum_asymmetry * 0.5  # Simplified early session concentration
        asymmetric_volume_pressure = pressure_asymmetry * 0.5  # Simplified late session momentum
        volume_confirmed_asymmetry = asymmetry_direction_consistency * volume_fractal_persistence
        
        fractal_volume_asymmetry = 0.5 * 0.5  # Simplified
        volume_microstructure_momentum = volume_fractal_persistence * (volume_t / (volume_t_1 + 1e-8))
        regime_volume_alignment = volume_regime_signature * volume_enhanced_asymmetry
        
        microstructure_asymmetry_composite = pressure_asymmetry * momentum_asymmetry * volume_enhanced_asymmetry
        fractal_asymmetry_momentum = multi_timeframe_alignment * asymmetry_magnitude_convergence
        volume_validated_asymmetry = asymmetric_volume_pressure * volume_confirmed_asymmetry
        
        # Adaptive Asymmetric Signal Synthesis
        high_fractal_asymmetry = microstructure_asymmetry_composite * high_fractal_regime
        volume_driven_asymmetry = fractal_asymmetry_momentum * volume_driven_regime
        price_led_asymmetry = volume_validated_asymmetry * price_led_regime
        
        ultra_short_fractal_asymmetry = ultra_short_asymmetry * fractal_compression_ratio
        short_term_regime_asymmetry = short_term_asymmetry * volume_regime_signature
        medium_term_fractal_momentum = medium_term_asymmetry * price_regime_fractal
        
        bidirectional_microstructure_momentum = pressure_asymmetry * closing_fractal_momentum * volume_microstructure_momentum
        fractal_asymmetry_alignment = asymmetry_direction_consistency * multi_timeframe_alignment * fractal_volume_asymmetry
        regime_volume_price_triad = volume_driven_regime * price_led_regime * high_fractal_regime
        
        # Cross-Fractal Validation Signals
        fractal_asymmetry_confirmation = high_fractal_asymmetry * volume_driven_asymmetry * price_led_asymmetry
        multi_regime_asymmetric_momentum = ultra_short_fractal_asymmetry * short_term_regime_asymmetry * medium_term_fractal_momentum
        integrated_microstructure_alpha = bidirectional_microstructure_momentum * fractal_asymmetry_alignment * regime_volume_price_triad
        
        # Final alpha factor
        result.iloc[i] = (
            fractal_asymmetry_confirmation + 
            multi_regime_asymmetric_momentum + 
            integrated_microstructure_alpha
        )
    
    # Fill initial NaN values with 0
    result = result.fillna(0)
    
    return result
