import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Momentum-Volume Resonance Factor
    Combines multi-timeframe momentum efficiency with asymmetric pressure patterns
    Integrates compression breakout detection with acceleration-weighted momentum
    Adapts to regime changes through pressure-resonance coupling
    """
    
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    for i in range(4, len(df)):
        current_data = df.iloc[:i+1].copy()
        
        # Multi-Scale Momentum Components
        if i >= 1:
            price_volume_momentum = (current_data['close'].iloc[i] - current_data['close'].iloc[i-1]) * current_data['volume'].iloc[i]
            volume_acceleration = (current_data['volume'].iloc[i] - current_data['volume'].iloc[i-1]) / current_data['volume'].iloc[i-1] if current_data['volume'].iloc[i-1] != 0 else 0
        else:
            price_volume_momentum = 0
            volume_acceleration = 0
            
        range_efficiency = (current_data['close'].iloc[i] - current_data['open'].iloc[i]) / (current_data['high'].iloc[i] - current_data['low'].iloc[i]) if (current_data['high'].iloc[i] - current_data['low'].iloc[i]) != 0 else 0
        
        # Resonance Dynamics
        micro_resonance = (current_data['high'].iloc[i] - current_data['low'].iloc[i]) * current_data['volume'].iloc[i] / current_data['amount'].iloc[i] if current_data['amount'].iloc[i] != 0 else 0
        
        # Meso-Resonance: Rolling price-volume correlation with triangular weighting
        window_size = min(5, i+1)
        if window_size >= 2:
            prices = current_data['close'].iloc[i-window_size+1:i+1].values
            volumes = current_data['volume'].iloc[i-window_size+1:i+1].values
            weights = np.array([j+1 for j in range(window_size)])
            weighted_corr = np.corrcoef(prices * weights, volumes * weights)[0,1] if len(prices) > 1 and np.std(prices * weights) > 0 and np.std(volumes * weights) > 0 else 0
            meso_resonance = weighted_corr
        else:
            meso_resonance = 0
        
        # Macro-Resonance: Cumulative momentum divergence
        if i >= 3:
            short_momentum = current_data['close'].iloc[i] - current_data['close'].iloc[i-1]
            medium_momentum = current_data['close'].iloc[i] - current_data['close'].iloc[i-2]
            long_momentum = current_data['close'].iloc[i] - current_data['close'].iloc[i-3]
            macro_resonance = (short_momentum + medium_momentum + long_momentum) / 3
        else:
            macro_resonance = 0
        
        # Momentum-Volume Synthesis
        momentum_volume_divergence = range_efficiency * (micro_resonance - macro_resonance)
        acceleration_weighted_momentum = volume_acceleration * price_volume_momentum
        resonance_regime_detection = abs(micro_resonance / macro_resonance - 1) * range_efficiency if macro_resonance != 0 else 0
        
        # Asymmetric Pressure-Resonance Patterns
        bull_pressure = (current_data['close'].iloc[i] - current_data['low'].iloc[i]) * current_data['volume'].iloc[i]
        bear_pressure = (current_data['high'].iloc[i] - current_data['close'].iloc[i]) * current_data['volume'].iloc[i]
        pressure_imbalance = bull_pressure / bear_pressure if bear_pressure != 0 else 1
        
        # Multi-Timeframe Resonance Integration
        if i >= 3:
            positive_momentum_volume = sum(current_data['volume'].iloc[j] for j in range(i-2, i+1) if current_data['close'].iloc[j] > current_data['open'].iloc[j])
            negative_momentum_volume = sum(current_data['volume'].iloc[j] for j in range(i-2, i+1) if current_data['close'].iloc[j] < current_data['open'].iloc[j])
            momentum_volume_ratio = positive_momentum_volume / negative_momentum_volume if negative_momentum_volume != 0 else 1
        else:
            momentum_volume_ratio = 1
        
        # Pressure-Resonance Coupling
        efficiency_pressure_alignment = range_efficiency * pressure_imbalance
        volume_resonance_divergence = volume_acceleration * momentum_volume_ratio
        nonlinear_microstructure_state = abs(range_efficiency - pressure_imbalance) * volume_acceleration
        
        # Momentum Compression Breakout Signals
        if i >= 4:
            recent_highs = current_data['high'].iloc[i-4:i+1]
            recent_lows = current_data['low'].iloc[i-4:i+1]
            range_contraction = (current_data['high'].iloc[i] - current_data['low'].iloc[i]) / (max(recent_highs) - min(recent_lows)) if (max(recent_highs) - min(recent_lows)) != 0 else 1
            
            compression_threshold = range_contraction < 0.6
            momentum_compression = abs(volume_acceleration) < 0.3
            
            # Asymmetric Breakout Detection
            price_breakout = current_data['close'].iloc[i] > max(current_data['high'].iloc[i-3:i])
            volume_surge = current_data['volume'].iloc[i] > 2 * current_data['volume'].iloc[i-1] if i >= 1 else False
            breakout_asymmetry = (current_data['close'].iloc[i] - current_data['low'].iloc[i]) / (current_data['high'].iloc[i] - current_data['close'].iloc[i]) if (current_data['high'].iloc[i] - current_data['close'].iloc[i]) != 0 else 1
            
            # Resonance-Enhanced Breakout
            resonance_breakout_alignment = int(price_breakout) * momentum_volume_divergence
            volume_efficiency_confirmation = int(volume_surge) * range_efficiency
            pressure_breakout_synchronization = pressure_imbalance * breakout_asymmetry
        else:
            range_contraction = 1
            compression_threshold = False
            momentum_compression = False
            price_breakout = False
            volume_surge = False
            breakout_asymmetry = 1
            resonance_breakout_alignment = 0
            volume_efficiency_confirmation = 0
            pressure_breakout_synchronization = 0
        
        # Composite Signal Generation
        momentum_resonance_factor = acceleration_weighted_momentum * efficiency_pressure_alignment
        asymmetric_breakout_power = breakout_asymmetry * range_efficiency
        regime_adaptive_scaling = nonlinear_microstructure_state * volume_acceleration
        
        # Divergence Enhancement
        multi_timeframe_convergence = momentum_volume_divergence * volume_resonance_divergence
        pressure_volume_alignment = np.sign(pressure_imbalance - 1) * np.sign(volume_acceleration - 0)
        resonance_breakout_filter = int(momentum_compression) * resonance_breakout_alignment
        
        # Cross-Validation Components
        if i >= 1:
            amount_intensity_adjustment = current_data['amount'].iloc[i] / current_data['amount'].iloc[i-1] if current_data['amount'].iloc[i-1] != 0 else 1
        else:
            amount_intensity_adjustment = 1
            
        sustained_momentum_confirmation = momentum_volume_ratio * volume_resonance_divergence
        volatility_state_filter = resonance_regime_detection * range_contraction
        
        # Final Alpha Factor Synthesis
        core_factor = momentum_resonance_factor * asymmetric_breakout_power
        regime_adjusted = core_factor * (1 + regime_adaptive_scaling)
        divergence_enhanced = regime_adjusted * (1 + multi_timeframe_convergence)
        
        # Apply filters and adjustments
        final_factor = divergence_enhanced * amount_intensity_adjustment * sustained_momentum_confirmation * (1 + volatility_state_filter)
        
        result.iloc[i] = final_factor
    
    # Fill initial NaN values with 0
    result = result.fillna(0)
    
    return result
