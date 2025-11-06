import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    for i in range(21, len(df)):
        current_data = df.iloc[:i+1].copy()
        
        # Quantum Fracture Momentum Divergence
        if i >= 3:
            close_t = current_data['close'].iloc[i]
            close_t_3 = current_data['close'].iloc[i-3]
            close_t_1 = current_data['close'].iloc[i-1]
            high_t = current_data['high'].iloc[i]
            low_t = current_data['low'].iloc[i]
            volume_t = current_data['volume'].iloc[i]
            volume_t_1 = current_data['volume'].iloc[i-1]
            
            qfmd_numerator = (close_t / close_t_3 - 1) * (high_t - low_t)
            qfmd_denominator = abs(close_t - close_t_1) * abs(volume_t - volume_t_1) / volume_t_1
            quantum_fracture_momentum_divergence = qfmd_numerator / (qfmd_denominator + 1e-8)
        else:
            quantum_fracture_momentum_divergence = 0
        
        # Multi-Frequency Momentum Fracture
        # Micro Momentum (1-day)
        if i >= 1:
            close_t_1 = current_data['close'].iloc[i-1]
            open_t = current_data['open'].iloc[i]
            high_t = current_data['high'].iloc[i]
            low_t = current_data['low'].iloc[i]
            volume_t = current_data['volume'].iloc[i]
            volume_t_1 = current_data['volume'].iloc[i-1]
            
            micro_momentum = ((close_t - close_t_1) / (high_t - low_t + 1e-8) * 
                            (volume_t / (volume_t_1 + 1e-8)) * 
                            (abs(close_t - open_t) / (abs(open_t - close_t_1) + 1e-8)))
        else:
            micro_momentum = 0
        
        # Meso Momentum (5-day)
        if i >= 5:
            close_t_5 = current_data['close'].iloc[i-5]
            volume_t_5 = current_data['volume'].iloc[i-5]
            high_window = current_data['high'].iloc[i-5:i+1]
            low_window = current_data['low'].iloc[i-5:i+1]
            
            meso_momentum = ((close_t - close_t_5) / (high_window.max() - low_window.min() + 1e-8) * 
                           (volume_t / (volume_t_5 + 1e-8)) * 
                           (abs(open_t - close_t_1) / (close_t_1 + 1e-8)))
        else:
            meso_momentum = 0
        
        # Macro Momentum (21-day)
        if i >= 21:
            close_t_21 = current_data['close'].iloc[i-21]
            volume_t_21 = current_data['volume'].iloc[i-21]
            amount_t = current_data['amount'].iloc[i]
            amount_t_1 = current_data['amount'].iloc[i-1]
            high_window_21 = current_data['high'].iloc[i-21:i+1]
            low_window_21 = current_data['low'].iloc[i-21:i+1]
            
            macro_momentum = ((close_t - close_t_21) / (high_window_21.max() - low_window_21.min() + 1e-8) * 
                            (volume_t / (volume_t_21 + 1e-8)) * 
                            (amount_t / (amount_t_1 + 1e-8) - 1))
        else:
            macro_momentum = 0
        
        # Quantum Momentum Alignment
        quantum_momentum_alignment = (quantum_fracture_momentum_divergence - 
                                    (micro_momentum + meso_momentum + macro_momentum) / 3 * 
                                    np.sign(volume_t - volume_t_1))
        
        # Quantum Gap and Efficiency Momentum
        if i >= 1:
            high_t_1 = current_data['high'].iloc[i-1]
            low_t_1 = current_data['low'].iloc[i-1]
            amount_t_1 = current_data['amount'].iloc[i-1]
            
            quantum_gap_momentum_efficiency = (abs(close_t - open_t) / (abs(open_t - close_t_1) + 1e-8) * 
                                            (high_t - low_t) / (high_t_1 - low_t_1 + 1e-8) * 
                                            (amount_t / (amount_t_1 + 1e-8) - 1))
            
            quantum_opening_momentum_efficiency = ((close_t - open_t) / (high_t - low_t + 1e-8) * 
                                                 (open_t - close_t_1) / (abs(open_t - close_t_1) + 1e-8) * 
                                                 volume_t / (volume_t_1 + 1e-8))
            
            quantum_gap_momentum_strength = ((close_t - low_t) / (high_t - low_t + 1e-8) * 
                                           abs(open_t - close_t_1) / (high_t - low_t + 1e-8) * 
                                           abs(volume_t - volume_t_1) / (volume_t_1 + 1e-8))
        else:
            quantum_gap_momentum_efficiency = 0
            quantum_opening_momentum_efficiency = 0
            quantum_gap_momentum_strength = 0
        
        # Quantum Volume-Momentum Dynamics
        if i >= 2:
            volume_t_2 = current_data['volume'].iloc[i-2]
            amount_t_2 = current_data['amount'].iloc[i-2]
            
            quantum_volume_momentum_ratio = (volume_t / (high_t - low_t + 1e-8) * 
                                           abs(close_t - open_t) / (abs(open_t - close_t_1) + 1e-8))
            
            quantum_amount_momentum_efficiency = (amount_t / (volume_t * close_t + 1e-8) * 
                                                (amount_t / (amount_t_1 + 1e-8) - 1))
            
            quantum_volume_momentum_acceleration = ((volume_t / (volume_t_2 + 1e-8) - 1) * 
                                                  np.sign(close_t - close_t_1) * 
                                                  (np.sign(volume_t - volume_t_2) != np.sign(close_t - close_t_1)) * 
                                                  abs(open_t - close_t_1) / (close_t_1 + 1e-8))
            
            quantum_trade_size_momentum = ((amount_t / (volume_t + 1e-8)) / (amount_t_1 / (volume_t_1 + 1e-8)) * 
                                         volume_t / (volume_t_1 + 1e-8))
        else:
            quantum_volume_momentum_ratio = 0
            quantum_amount_momentum_efficiency = 0
            quantum_volume_momentum_acceleration = 0
            quantum_trade_size_momentum = 0
        
        # Quantum Momentum Regime Adaptation
        if i >= 15:
            high_t_3 = current_data['high'].iloc[i-3]
            low_t_3 = current_data['low'].iloc[i-3]
            high_t_15 = current_data['high'].iloc[i-15]
            low_t_15 = current_data['low'].iloc[i-15]
            
            price_range_ratio = (high_t - low_t) / (close_t_1 + 1e-8)
            volume_change_ratio = abs(volume_t - volume_t_1) / (volume_t_1 + 1e-8)
            
            # Regime classification
            if price_range_ratio > 0.04 and volume_change_ratio > 0.3:
                regime_weighted_momentum = (micro_momentum * 0.7 + 
                                          meso_momentum * 0.3 * volume_change_ratio)
            elif price_range_ratio < 0.01 and volume_change_ratio < 0.1:
                regime_weighted_momentum = (meso_momentum * 0.6 + 
                                          macro_momentum * 0.4 * (amount_t / (amount_t_1 + 1e-8) - 1))
            else:
                regime_weighted_momentum = ((micro_momentum + meso_momentum + macro_momentum) / 3 * 
                                          volume_t / (volume_t_1 + 1e-8))
            
            quantum_momentum_regime_shift = ((high_t - low_t) / (high_t_3 - low_t_3 + 1e-8) - 
                                           (high_t - low_t) / (high_t_15 - low_t_15 + 1e-8) * 
                                           volume_change_ratio)
        else:
            regime_weighted_momentum = 0
            quantum_momentum_regime_shift = 0
        
        # Quantum Momentum Entanglement
        if i >= 2:
            close_t_2 = current_data['close'].iloc[i-2]
            high_t_1 = current_data['high'].iloc[i-1]
            low_t_1 = current_data['low'].iloc[i-1]
            
            # Opening Momentum Entanglement
            gap_momentum_fracture = (np.sign(open_t - close_t_1) * 
                                   abs(open_t - close_t_1) / (high_t - low_t + 1e-8) * 
                                   abs(close_t_1 - close_t_2) / (high_t_1 - low_t_1 + 1e-8))
            
            opening_momentum_efficiency_ent = ((close_t - open_t) * np.sign(open_t - close_t_1) / 
                                             (high_t - low_t + 1e-8) * 
                                             abs(close_t_1 - close_t_2) / (high_t_1 - low_t_1 + 1e-8))
            
            # Intraday Momentum Entanglement
            upper_momentum_efficiency = ((high_t - close_t) / (high_t - open_t + 1e-8) * 
                                       abs(close_t - open_t) / (high_t - low_t + 1e-8) * 
                                       abs(close_t_1 - close_t_2) / (high_t_1 - low_t_1 + 1e-8))
            
            lower_momentum_efficiency = ((close_t - low_t) / (open_t - low_t + 1e-8) * 
                                       abs(close_t - open_t) / (high_t - low_t + 1e-8) * 
                                       abs(close_t_1 - close_t_2) / (high_t_1 - low_t_1 + 1e-8))
            
            # Closing Momentum Entanglement
            final_hour_momentum = ((close_t - (high_t + low_t) / 2) * 
                                 abs(close_t - open_t) / (high_t - low_t + 1e-8) * 
                                 abs(close_t_1 - close_t_2) / (high_t_1 - low_t_1 + 1e-8))
            
            overnight_momentum_reversal = ((np.sign(close_t - open_t) != np.sign(open_t - close_t_1)) * 
                                         abs(close_t - open_t) / (high_t - low_t + 1e-8) * 
                                         abs(close_t_1 - close_t_2) / (high_t_1 - low_t_1 + 1e-8))
        else:
            gap_momentum_fracture = 0
            opening_momentum_efficiency_ent = 0
            upper_momentum_efficiency = 0
            lower_momentum_efficiency = 0
            final_hour_momentum = 0
            overnight_momentum_reversal = 0
        
        # Quantum Momentum Breakout Detection
        quantum_volume_momentum_breakout = (volume_t > 1.5 * volume_t_1 and 
                                          quantum_gap_momentum_strength > 0.3 * abs(open_t - close_t_1) / (close_t_1 + 1e-8))
        
        quantum_efficiency_momentum_breakout = (quantum_amount_momentum_efficiency > 1.5 and 
                                              quantum_opening_momentum_efficiency > 0.6 * volume_t / (volume_t_1 + 1e-8))
        
        quantum_momentum_breakout_signal = (regime_weighted_momentum * 
                                          (1 + quantum_volume_momentum_breakout + quantum_efficiency_momentum_breakout) * 
                                          abs(volume_t - volume_t_1) / (volume_t_1 + 1e-8))
        
        # Quantum Momentum Asymmetry Framework
        quantum_momentum_convergence = (quantum_opening_momentum_efficiency * quantum_gap_momentum_efficiency * 
                                      (close_t - close_t_1) * np.sign(volume_t - volume_t_1))
        
        quantum_momentum_gradient = ((quantum_opening_momentum_efficiency - quantum_gap_momentum_efficiency) * 
                                   (quantum_gap_momentum_efficiency - quantum_amount_momentum_efficiency) * 
                                   volume_t * (amount_t / (amount_t_1 + 1e-8) - 1))
        
        quantum_momentum_divergence = (abs(quantum_opening_momentum_efficiency - quantum_gap_momentum_efficiency) / 
                                     (quantum_opening_momentum_efficiency + quantum_gap_momentum_efficiency + 1e-8) * 
                                     (volume_t - volume_t_1) * np.sign(close_t - close_t_1))
        
        # Quantum Composite Momentum Construction
        base_quantum_momentum = regime_weighted_momentum
        
        quantum_momentum_enhanced = base_quantum_momentum * (1 + quantum_momentum_alignment * 
                                                           (gap_momentum_fracture + opening_momentum_efficiency_ent))
        
        quantum_gap_momentum_enhanced = quantum_momentum_enhanced * (1 + quantum_gap_momentum_efficiency * 
                                                                   quantum_gap_momentum_strength)
        
        quantum_volume_momentum_enhanced = quantum_gap_momentum_enhanced * (1 + quantum_volume_momentum_ratio * 
                                                                          quantum_amount_momentum_efficiency)
        
        quantum_volume_aligned = quantum_volume_momentum_enhanced * quantum_volume_momentum_acceleration
        
        quantum_final_adjustment = quantum_volume_aligned * (1 + quantum_momentum_divergence) * quantum_trade_size_momentum
        
        # Quantum Hierarchical Momentum Alpha
        # Base Momentum Signal
        primary_momentum = (gap_momentum_fracture * upper_momentum_efficiency * final_hour_momentum)
        secondary_momentum = (quantum_momentum_breakout_signal * quantum_momentum_regime_shift)
        
        # Momentum Regime Weighting
        range_momentum_weight = quantum_momentum_regime_shift
        volume_momentum_weight = quantum_volume_momentum_breakout + quantum_efficiency_momentum_breakout
        efficiency_momentum_weight = quantum_opening_momentum_efficiency + quantum_gap_momentum_efficiency
        
        momentum_regime_weighting = range_momentum_weight + volume_momentum_weight + efficiency_momentum_weight
        
        # Quantum Momentum Convergence
        bullish_momentum = (quantum_momentum_convergence * quantum_volume_momentum_acceleration * final_hour_momentum)
        bearish_momentum = (quantum_momentum_divergence * overnight_momentum_reversal * quantum_volume_momentum_ratio)
        
        # Final Quantum Hierarchical Momentum Alpha
        base_alpha = primary_momentum * (1 + momentum_regime_weighting)
        quantum_adjustment = (quantum_momentum_gradient * quantum_volume_momentum_ratio * 
                            (quantum_momentum_convergence + quantum_momentum_divergence))
        
        final_alpha = (base_alpha * (1 + quantum_adjustment) * 
                     (bullish_momentum - bearish_momentum) * secondary_momentum)
        
        result.iloc[i] = final_alpha
    
    # Fill NaN values with 0
    result = result.fillna(0)
    
    return result
