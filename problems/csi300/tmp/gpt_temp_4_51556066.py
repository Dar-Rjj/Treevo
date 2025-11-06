import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate required components with proper shifting to avoid lookahead bias
    for i in range(10, len(df)):
        current_data = df.iloc[:i+1]  # Only use data up to current day
        
        # Gap-Pressure Momentum Components
        if i >= 1:
            gap_momentum_score = ((current_data['open'].iloc[i] / current_data['close'].iloc[i-1] - 1) * 
                                (current_data['close'].iloc[i] - current_data['open'].iloc[i]) / 
                                (current_data['close'].iloc[i-1] - current_data['open'].iloc[i-1]))
            
            gap_pressure_magnitude = ((current_data['open'].iloc[i] / current_data['close'].iloc[i-1] - 1) * 
                                    current_data['volume'].iloc[i])
            
            gap_filling_pressure = ((current_data['close'].iloc[i] - current_data['open'].iloc[i]) / 
                                  (current_data['close'].iloc[i-1] - current_data['open'].iloc[i-1]) * 
                                  current_data['amount'].iloc[i])
        else:
            gap_momentum_score = gap_pressure_magnitude = gap_filling_pressure = 0
        
        # Fractal Gap Dynamics
        if i >= 2:
            fractal_range_ratio = ((current_data['high'].iloc[i] - current_data['low'].iloc[i]) / 
                                 (current_data['high'].iloc[i-2] - current_data['low'].iloc[i-2]))
            
            volatility_adjusted_gap_pressure = (gap_pressure_magnitude / 
                                              (current_data['high'].iloc[i] - current_data['low'].iloc[i]))
        else:
            fractal_range_ratio = volatility_adjusted_gap_pressure = 1
        
        if i >= 6:
            gap_momentum_persistence = (((current_data['close'].iloc[i] / current_data['close'].iloc[i-3] - 1) / 
                                       (current_data['close'].iloc[i] / current_data['close'].iloc[i-6] - 1)) * 
                                      np.sign(current_data['close'].iloc[i] / current_data['close'].iloc[i-1] - 1))
        else:
            gap_momentum_persistence = 1
        
        # Gap-Pressure Alignment
        gap_pressure_efficiency = (gap_filling_pressure * current_data['amount'].iloc[i] / 
                                 (current_data['high'].iloc[i] - current_data['low'].iloc[i]))
        
        gap_momentum_confirmation = gap_momentum_score * gap_momentum_persistence
        fractal_gap_integration = gap_pressure_magnitude * fractal_range_ratio
        
        # Asymmetry-Efficiency Dynamics
        intraday_asymmetry_pressure = ((current_data['close'].iloc[i] - current_data['open'].iloc[i]) / 
                                     (current_data['high'].iloc[i] - current_data['low'].iloc[i]) * 
                                     current_data['volume'].iloc[i])
        
        high_open_diff = current_data['high'].iloc[i] - current_data['open'].iloc[i]
        open_low_diff = current_data['open'].iloc[i] - current_data['low'].iloc[i]
        close_low_diff = current_data['close'].iloc[i] - current_data['low'].iloc[i]
        high_close_diff = current_data['high'].iloc[i] - current_data['close'].iloc[i]
        
        intraday_efficiency_asymmetry = ((high_open_diff / open_low_diff) if open_low_diff != 0 else 0) - \
                                      ((close_low_diff / high_close_diff) if high_close_diff != 0 else 0)
        
        micro_momentum_pressure = intraday_efficiency_asymmetry * current_data['volume'].iloc[i]
        
        # Multi-scale Asymmetry Pressure
        if i >= 2:
            sum_high_open_3d = sum(current_data['high'].iloc[i-2:i+1] - current_data['open'].iloc[i-2:i+1])
            sum_open_low_3d = sum(current_data['open'].iloc[i-2:i+1] - current_data['low'].iloc[i-2:i+1])
            avg_volume_3d = current_data['volume'].iloc[i-2:i+1].mean()
            asymmetry_pressure_3d = (sum_high_open_3d / sum_open_low_3d if sum_open_low_3d != 0 else 0) * avg_volume_3d
        else:
            asymmetry_pressure_3d = 0
        
        if i >= 4:
            sum_high_open_5d = sum(current_data['high'].iloc[i-4:i+1] - current_data['open'].iloc[i-4:i+1])
            sum_open_low_5d = sum(current_data['open'].iloc[i-4:i+1] - current_data['low'].iloc[i-4:i+1])
            avg_volume_5d = current_data['volume'].iloc[i-4:i+1].mean()
            asymmetry_pressure_5d = (sum_high_open_5d / sum_open_low_5d if sum_open_low_5d != 0 else 0) * avg_volume_5d
        else:
            asymmetry_pressure_5d = 0
        
        # Efficiency-Pressure Integration
        if i >= 1:
            fractal_efficiency = (abs(current_data['close'].iloc[i] - current_data['close'].iloc[i-1]) / 
                               (current_data['high'].iloc[i] - current_data['low'].iloc[i]) * fractal_range_ratio)
            
            volume_pressure_efficiency = (current_data['amount'].iloc[i] / 
                                       abs(current_data['close'].iloc[i] - current_data['close'].iloc[i-1]) * 
                                       np.sign(current_data['close'].iloc[i] - current_data['close'].iloc[i-1]))
        else:
            fractal_efficiency = volume_pressure_efficiency = 0
        
        amount_concentration_efficiency = (current_data['amount'].iloc[i] / 
                                         (current_data['high'].iloc[i] - current_data['low'].iloc[i]) * 
                                         abs(current_data['close'].iloc[i] - current_data['open'].iloc[i]))
        
        # Volume-Pressure Dynamics
        if i >= 5:
            volume_pressure_score = (((current_data['volume'].iloc[i] / current_data['volume'].iloc[i-3] - 1) / 
                                   (current_data['volume'].iloc[i] / current_data['volume'].iloc[i-5] - 1)) * 
                                  np.sign(current_data['volume'].iloc[i] / current_data['volume'].iloc[i-3] - 1))
        else:
            volume_pressure_score = 0
        
        if i >= 2:
            volume_spike_pressure = (current_data['volume'].iloc[i] / 
                                   current_data['volume'].iloc[i-2:i+1].mean() * 
                                   (current_data['high'].iloc[i] - current_data['low'].iloc[i]))
        else:
            volume_spike_pressure = 0
        
        if i >= 1:
            amount_persistence_pressure = (current_data['amount'].iloc[i] / current_data['amount'].iloc[i-1] * 
                                         np.sign(current_data['close'].iloc[i] - current_data['close'].iloc[i-1]))
        else:
            amount_persistence_pressure = 0
        
        # Price-Volume Divergence
        if i >= 3:
            price_volume_divergence = (((current_data['close'].iloc[i] / current_data['close'].iloc[i-3] - 1) / 
                                      (current_data['volume'].iloc[i] / current_data['volume'].iloc[i-3] - 1)) * 
                                     np.sign(current_data['close'].iloc[i] / current_data['close'].iloc[i-1] - 1))
        else:
            price_volume_divergence = 0
        
        if i >= 1:
            liquidity_pressure = (((current_data['high'].iloc[i] + current_data['low'].iloc[i] - 
                                 2 * current_data['close'].iloc[i-1]) / 
                                (current_data['high'].iloc[i] - current_data['low'].iloc[i])) * 
                               (current_data['volume'].iloc[i] / current_data['volume'].iloc[i-1]))
        else:
            liquidity_pressure = 0
        
        if i >= 1:
            volume_velocity_pressure = (current_data['amount'].iloc[i] / current_data['volume'].iloc[i] * 
                                      abs(current_data['close'].iloc[i] - current_data['close'].iloc[i-1]))
        else:
            volume_velocity_pressure = 0
        
        # Pressure Alignment Components
        volume_pressure_confirmation = volume_pressure_score * volume_spike_pressure
        price_volume_alignment = price_volume_divergence * liquidity_pressure
        efficiency_pressure_integration = volume_pressure_efficiency * amount_concentration_efficiency
        
        # Breakout-Pressure Integration
        if i >= 1:
            net_breakout_velocity = (((current_data['high'].iloc[i] - current_data['close'].iloc[i-1]) - 
                                    (current_data['close'].iloc[i-1] - current_data['low'].iloc[i])) * 
                                   current_data['volume'].iloc[i])
            
            if i >= 5:
                net_breakout_velocity *= (current_data['close'].iloc[i] / current_data['close'].iloc[i-5] - 1 - 
                                        (current_data['close'].iloc[i] / current_data['close'].iloc[i-10] - 1) if i >= 10 else 0)
            
            upside_breakout_pressure = ((current_data['high'].iloc[i] - current_data['close'].iloc[i-1]) * 
                                      current_data['volume'].iloc[i] * current_data['amount'].iloc[i])
            
            downside_breakout_pressure = ((current_data['close'].iloc[i-1] - current_data['low'].iloc[i]) * 
                                        current_data['volume'].iloc[i] * current_data['amount'].iloc[i])
            
            net_breakout_pressure = upside_breakout_pressure - downside_breakout_pressure
        else:
            net_breakout_velocity = upside_breakout_pressure = downside_breakout_pressure = net_breakout_pressure = 0
        
        # Range-Pressure Dynamics
        if i >= 1:
            true_range = max(current_data['high'].iloc[i] - current_data['low'].iloc[i],
                           abs(current_data['high'].iloc[i] - current_data['close'].iloc[i-1]),
                           abs(current_data['low'].iloc[i] - current_data['close'].iloc[i-1]))
            
            true_range_pressure = true_range * current_data['volume'].iloc[i]
        else:
            true_range_pressure = 0
        
        if i >= 5:
            range_constriction_pressure = ((current_data['high'].iloc[i] - current_data['low'].iloc[i]) / 
                                         (current_data['high'].iloc[i-5] - current_data['low'].iloc[i-5]) * 
                                         current_data['volume'].iloc[i])
        else:
            range_constriction_pressure = 0
        
        if i >= 1:
            range_momentum_pressure = (abs(current_data['close'].iloc[i] - current_data['close'].iloc[i-1]) / 
                                     true_range * current_data['amount'].iloc[i] * 
                                     np.sign(current_data['close'].iloc[i] - current_data['close'].iloc[i-1]))
        else:
            range_momentum_pressure = 0
        
        # Breakout-Efficiency Alignment
        breakout_pressure_efficiency = net_breakout_pressure * amount_concentration_efficiency
        
        if i >= 5:
            range_velocity_pressure = range_momentum_pressure * (current_data['close'].iloc[i] / current_data['close'].iloc[i-5] - 1)
        else:
            range_velocity_pressure = 0
        
        fractal_breakout_integration = net_breakout_velocity * fractal_range_ratio
        
        # Multi-timeframe Momentum Confirmation
        if i >= 2 and i >= 1:
            short_term_momentum = ((current_data['close'].iloc[i] / current_data['close'].iloc[i-2] - 1) / 
                                 (current_data['close'].iloc[i] / current_data['close'].iloc[i-1] - 1))
        else:
            short_term_momentum = 0
        
        if i >= 5:
            price_velocity_pressure = ((current_data['close'].iloc[i] / current_data['close'].iloc[i-5] - 1) * 
                                     current_data['volume'].iloc[i])
        else:
            price_velocity_pressure = 0
        
        micro_momentum_alignment = intraday_asymmetry_pressure * short_term_momentum
        
        if i >= 5 and i >= 3:
            medium_term_momentum = ((current_data['close'].iloc[i] / current_data['close'].iloc[i-5] - 1) / 
                                  (current_data['close'].iloc[i] / current_data['close'].iloc[i-3] - 1))
        else:
            medium_term_momentum = 0
        
        if i >= 6:
            momentum_persistence = (((current_data['close'].iloc[i] / current_data['close'].iloc[i-3] - 1) / 
                                  (current_data['close'].iloc[i] / current_data['close'].iloc[i-6] - 1)) * 
                                 np.sign(current_data['close'].iloc[i] / current_data['close'].iloc[i-1] - 1))
        else:
            momentum_persistence = 0
        
        meso_momentum_alignment = asymmetry_pressure_3d * medium_term_momentum
        
        # Momentum Convergence Dynamics
        momentum_convergence = short_term_momentum * medium_term_momentum * np.sign(short_term_momentum - medium_term_momentum)
        multi_scale_momentum_integration = micro_momentum_alignment * meso_momentum_alignment
        velocity_pressure_alignment = price_velocity_pressure * volume_velocity_pressure
        
        # Base Momentum Core
        gap_pressure_momentum = gap_momentum_score * gap_filling_pressure * gap_momentum_persistence
        asymmetry_efficiency_core = intraday_efficiency_asymmetry * fractal_efficiency * volume_pressure_efficiency
        breakout_pressure_core = net_breakout_velocity * range_momentum_pressure * breakout_pressure_efficiency
        
        # Multi-scale Integration
        short_term_pressure_alpha = ((gap_pressure_magnitude * volume_pressure_efficiency) / 
                                   true_range_pressure if true_range_pressure != 0 else 0)
        
        if i >= 4:
            avg_true_range_5d = sum([max(current_data['high'].iloc[j] - current_data['low'].iloc[j],
                                       abs(current_data['high'].iloc[j] - current_data['close'].iloc[j-1]),
                                       abs(current_data['low'].iloc[j] - current_data['close'].iloc[j-1])) 
                                   for j in range(i-4, i+1)]) / 5
            medium_term_pressure_alpha = ((asymmetry_pressure_5d * volume_pressure_confirmation) / 
                                        avg_true_range_5d if avg_true_range_5d != 0 else 0)
        else:
            medium_term_pressure_alpha = 0
        
        divergence_pressure_alpha = price_volume_divergence * momentum_convergence * volume_pressure_efficiency
        
        # Adaptive Core Construction
        core_pressure_factor = gap_pressure_momentum * asymmetry_efficiency_core * breakout_pressure_core
        pressure_release_adjustment = core_pressure_factor * (range_constriction_pressure * volume_spike_pressure)
        volume_pressure_confirmation_final = pressure_release_adjustment * volume_pressure_score
        
        # Final Alpha Output
        directional_momentum_alignment = volume_pressure_confirmation_final * price_volume_alignment
        multi_scale_fractal_alpha = (short_term_pressure_alpha + medium_term_pressure_alpha + 
                                   divergence_pressure_alpha + directional_momentum_alignment)
        
        result.iloc[i] = multi_scale_fractal_alpha
    
    # Fill early NaN values with 0
    result = result.fillna(0)
    
    return result
