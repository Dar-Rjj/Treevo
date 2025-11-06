import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df)):
        if i < 20:  # Need at least 20 periods for calculations
            result.iloc[i] = 0
            continue
            
        # Current values
        open_t = df.iloc[i]['open']
        high_t = df.iloc[i]['high']
        low_t = df.iloc[i]['low']
        close_t = df.iloc[i]['close']
        volume_t = df.iloc[i]['volume']
        amount_t = df.iloc[i]['amount']
        
        # Previous values
        close_t_1 = df.iloc[i-1]['close']
        close_t_2 = df.iloc[i-2]['close']
        close_t_5 = df.iloc[i-5]['close']
        close_t_10 = df.iloc[i-10]['close']
        close_t_20 = df.iloc[i-20]['close']
        
        volume_t_1 = df.iloc[i-1]['volume']
        volume_t_2 = df.iloc[i-2]['volume']
        volume_t_5 = df.iloc[i-5]['volume']
        volume_t_20 = df.iloc[i-20]['volume']
        
        high_t_1 = df.iloc[i-1]['high']
        low_t_1 = df.iloc[i-1]['low']
        open_t_1 = df.iloc[i-1]['open']
        
        # Multi-Frequency Regime Detection
        # High-frequency regime
        if high_t != low_t and close_t_1 != 0:
            hf_regime = ((close_t - open_t) / (high_t - low_t) * 
                        (volume_t / volume_t_1) * 
                        (abs(open_t - close_t_1) / close_t_1) * 
                        np.sign(close_t - open_t))
        else:
            hf_regime = 0
            
        # Medium-frequency regime
        vol_5 = sum(abs(df.iloc[j]['close'] - df.iloc[j-1]['close']) for j in range(i-4, i+1))
        if vol_5 != 0 and abs(open_t - close_t_1) != 0:
            mf_regime = ((close_t - close_t_5) / vol_5 * 
                        (high_t - low_t) / abs(open_t - close_t_1) * 
                        np.sign(close_t - open_t))
        else:
            mf_regime = 0
            
        # Low-frequency regime
        vol_20 = sum(abs(df.iloc[j]['close'] - df.iloc[j-1]['close']) for j in range(i-19, i+1))
        if vol_20 != 0 and amount_t != 0 and abs(close_t - close_t_20) != 0:
            lf_regime = ((close_t - close_t_20) / vol_20 * 
                        abs(open_t - close_t_1) / abs(close_t - close_t_20) * 
                        volume_t / amount_t * 
                        np.sign(close_t - open_t))
        else:
            lf_regime = 0
            
        frequency_core = hf_regime * mf_regime * lf_regime
        
        # Asymmetric Microstructure Patterns
        # Volume Asymmetry
        if (volume_t + volume_t_1) != 0 and (high_t - low_t) != 0:
            volume_asymmetry = ((volume_t - volume_t_1) / (volume_t + volume_t_1) * 
                              (close_t - open_t) / (high_t - low_t))
        else:
            volume_asymmetry = 0
            
        # Price Position Asymmetry
        if (high_t - low_t) != 0:
            price_position_asymmetry = ((close_t - low_t) / (high_t - low_t) * 
                                      (high_t - close_t) / (high_t - low_t))
        else:
            price_position_asymmetry = 0
            
        # Upside Microstructure
        if (high_t - low_t) != 0:
            upside_micro = ((close_t - open_t) * (close_t - low_t) / 
                          ((high_t - low_t) ** 2) * volume_t)
        else:
            upside_micro = 0
            
        # Downside Microstructure
        if (high_t - low_t) != 0:
            downside_micro = ((open_t - close_t) * (high_t - close_t) / 
                            ((high_t - low_t) ** 2) * volume_t)
        else:
            downside_micro = 0
            
        # Asymmetric Balance
        if (upside_micro + downside_micro) != 0:
            asymmetric_balance = (upside_micro - downside_micro) / (upside_micro + downside_micro)
        else:
            asymmetric_balance = 0
            
        microstructure_core = volume_asymmetry * price_position_asymmetry * asymmetric_balance
        
        # Chaotic Fractal Dynamics
        # Fractal Efficiency
        if (high_t - low_t) > 0 and abs(close_t - close_t_1) > 0:
            fractal_efficiency = (np.log(high_t - low_t) / np.log(abs(close_t - close_t_1)) * 
                               volume_t / abs(close_t - close_t_1))
        else:
            fractal_efficiency = 0
            
        # Phase Space Momentum
        if (high_t - low_t) != 0:
            phase_space_momentum = ((close_t - close_t_1) * (close_t_1 - close_t_2) / 
                                  ((high_t - low_t) ** 2))
        else:
            phase_space_momentum = 0
            
        # Critical Transition
        if volume_t_1 != 0 and (high_t_1 - low_t_1) != 0:
            critical_transition = ((volume_t / volume_t_1 - 1) * 
                                 (high_t - low_t) / (high_t_1 - low_t_1))
        else:
            critical_transition = 0
            
        # Chaotic Attractor
        if (high_t - low_t) != 0:
            chaotic_attractor = (close_t - (high_t + low_t) / 2) / (high_t - low_t)
        else:
            chaotic_attractor = 0
            
        chaotic_core = fractal_efficiency * phase_space_momentum * critical_transition
        
        # Multi-Timeframe Pressure Dynamics
        # Short-Term Pressure
        vol_2 = sum(abs(df.iloc[j]['close'] - df.iloc[j-1]['close']) for j in range(i-1, i+1))
        if vol_2 != 0 and volume_t_2 != 0:
            short_term_pressure = ((close_t - close_t_2) / vol_2 * 
                                 volume_t / volume_t_2)
        else:
            short_term_pressure = 0
            
        # Medium-Term Pressure
        vol_10 = sum(abs(df.iloc[j]['close'] - df.iloc[j-1]['close']) for j in range(i-9, i+1))
        if vol_10 != 0 and abs(open_t - close_t_1) != 0:
            medium_term_pressure = ((close_t - close_t_10) / vol_10 * 
                                  (high_t - low_t) / abs(open_t - close_t_1))
        else:
            medium_term_pressure = 0
            
        # Long-Term Pressure
        if vol_20 != 0 and volume_t_20 != 0:
            long_term_pressure = ((close_t - close_t_20) / vol_20 * 
                                volume_t / volume_t_20)
        else:
            long_term_pressure = 0
            
        pressure_core = short_term_pressure * medium_term_pressure * long_term_pressure
        
        # Regime Transition Framework
        # Volume Regime Shift
        if volume_t_1 != 0 and volume_t_2 != 0:
            volume_regime_shift = (volume_t / volume_t_1) - (volume_t_1 / volume_t_2)
        else:
            volume_regime_shift = 0
            
        # Price Regime Momentum
        if (high_t - low_t) != 0 and (high_t_1 - low_t_1) != 0:
            price_regime_momentum = ((close_t - close_t_1) / (high_t - low_t) - 
                                   (close_t_1 - close_t_2) / (high_t_1 - low_t_1))
        else:
            price_regime_momentum = 0
            
        # Efficiency Regime Change
        if (high_t - low_t) != 0 and volume_t_1 != 0 and (high_t_1 - low_t_1) != 0 and volume_t_2 != 0:
            current_efficiency = (abs(close_t - open_t) / (high_t - low_t) * volume_t / volume_t_1)
            prev_efficiency = (abs(close_t_1 - open_t_1) / (high_t_1 - low_t_1) * volume_t_1 / volume_t_2)
            efficiency_regime_change = current_efficiency - prev_efficiency
        else:
            efficiency_regime_change = 0
            
        transition_multiplier = volume_regime_shift * price_regime_momentum * efficiency_regime_change
        
        # Hierarchical Alpha Construction
        base_alpha = frequency_core * microstructure_core * chaotic_core
        pressure_adjustment = base_alpha * pressure_core
        transition_enhancement = pressure_adjustment * (1 + transition_multiplier)
        final_alpha = transition_enhancement * chaotic_attractor
        
        # Risk-Aware Refinement
        # Volatility Adjustment
        if (high_t - low_t) != 0:
            volatility_adjusted = final_alpha / (high_t - low_t)
        else:
            volatility_adjusted = final_alpha
            
        # Volume Stability
        if volume_t_5 != 0:
            volume_stable = volatility_adjusted * volume_t / volume_t_5
        else:
            volume_stable = volatility_adjusted
            
        # Price Stability
        if (high_t - low_t) != 0:
            price_stable = volume_stable * abs(close_t - close_t_1) / (high_t - low_t)
        else:
            price_stable = volume_stable
            
        result.iloc[i] = price_stable
    
    return result
