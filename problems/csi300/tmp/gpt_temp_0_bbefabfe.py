import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate required components with proper shifting to avoid lookahead
    for i in range(len(df)):
        if i < 12:  # Need at least 12 days of history
            result.iloc[i] = 0
            continue
            
        # Extract current and historical data
        current = df.iloc[i]
        prev_1 = df.iloc[i-1] if i >= 1 else None
        prev_2 = df.iloc[i-2] if i >= 2 else None
        prev_3 = df.iloc[i-3] if i >= 3 else None
        prev_4 = df.iloc[i-4] if i >= 4 else None
        prev_5 = df.iloc[i-5] if i >= 5 else None
        prev_6 = df.iloc[i-6] if i >= 6 else None
        prev_10 = df.iloc[i-10] if i >= 10 else None
        
        # Price Pressure Components
        micro_pressure = ((current['close'] - prev_1['close']) / 
                         (current['high'] - current['low'] + 1e-8) * 
                         (current['amount'] / (prev_1['amount'] + 1e-8)))
        
        # Meso pressure (5-day window)
        high_5d = max(df.iloc[i-5:i+1]['high'])
        low_5d = min(df.iloc[i-5:i+1]['low'])
        meso_pressure = ((current['close'] - prev_5['close']) / 
                        (high_5d - low_5d + 1e-8) * 
                        (current['amount'] / (prev_5['amount'] + 1e-8)))
        
        # Macro pressure (10-day window)
        high_10d = max(df.iloc[i-10:i+1]['high'])
        low_10d = min(df.iloc[i-10:i+1]['low'])
        macro_pressure = ((current['close'] - prev_10['close']) / 
                         (high_10d - low_10d + 1e-8) * 
                         (current['amount'] / (prev_10['amount'] + 1e-8)))
        
        # Volume Pressure Dynamics
        volume_accel = ((current['volume'] / (prev_1['volume'] + 1e-8)) - 
                       (prev_1['volume'] / (prev_2['volume'] + 1e-8)))
        
        amount_conc = ((current['amount'] / (current['volume'] + 1e-8)) / 
                      (prev_1['amount'] / (prev_1['volume'] + 1e-8) + 1e-8))
        
        pressure_efficiency = (((current['close'] - current['open']) / 
                              (current['high'] - current['low'] + 1e-8)) * 
                              volume_accel)
        
        # Pressure Integration
        micro_pressure_int = micro_pressure * amount_conc
        meso_pressure_int = meso_pressure * volume_accel
        macro_pressure_int = macro_pressure * pressure_efficiency
        
        # Fractal Momentum Classification
        fractal_up = (current['close'] > prev_1['close'] and 
                     prev_1['close'] > prev_2['close'])
        fractal_down = (current['close'] < prev_1['close'] and 
                       prev_1['close'] < prev_2['close'])
        fractal_transition = not (fractal_up or fractal_down)
        
        range_expansion = (current['high'] - current['low']) > (prev_1['high'] - prev_1['low'])
        range_contraction = (current['high'] - current['low']) < (prev_1['high'] - prev_1['low'])
        
        expanding_up = fractal_up and range_expansion
        contracting_down = fractal_down and range_contraction
        mixed_fractal = not (expanding_up or contracting_down)
        
        # Gradient-Flow Mechanics
        positive_gradient = ((current['close'] - current['low']) / 
                           (current['high'] - current['low'] + 1e-8) * 
                           current['amount'])
        negative_gradient = ((current['high'] - current['close']) / 
                           (current['high'] - current['low'] + 1e-8) * 
                           current['amount'])
        net_gradient_flow = positive_gradient - negative_gradient
        
        # Medium Gradient (5-day window)
        net_gradients = []
        for j in range(max(0, i-4), i+1):
            if j >= 4:
                pos_grad = ((df.iloc[j]['close'] - df.iloc[j]['low']) / 
                          (df.iloc[j]['high'] - df.iloc[j]['low'] + 1e-8) * 
                          df.iloc[j]['amount'])
                neg_grad = ((df.iloc[j]['high'] - df.iloc[j]['close']) / 
                          (df.iloc[j]['high'] - df.iloc[j]['low'] + 1e-8) * 
                          df.iloc[j]['amount'])
                net_grad = pos_grad - neg_grad
                net_gradients.append(net_grad)
        
        medium_gradient = (sum(1 for ng in net_gradients if ng > 0) - 
                          sum(1 for ng in net_gradients if ng < 0)) if net_gradients else 0
        
        # Gradient Momentum
        if i >= 2:
            pos_grad_prev2 = ((df.iloc[i-2]['close'] - df.iloc[i-2]['low']) / 
                            (df.iloc[i-2]['high'] - df.iloc[i-2]['low'] + 1e-8) * 
                            df.iloc[i-2]['amount'])
            neg_grad_prev2 = ((df.iloc[i-2]['high'] - df.iloc[i-2]['close']) / 
                            (df.iloc[i-2]['high'] - df.iloc[i-2]['low'] + 1e-8) * 
                            df.iloc[i-2]['amount'])
            net_grad_prev2 = pos_grad_prev2 - neg_grad_prev2
            gradient_momentum = net_gradient_flow / (net_grad_prev2 + 1e-8)
        else:
            gradient_momentum = 0
        
        # Fractal-Gradient Integration
        fractal_scaled_gradient = ((current['high'] - current['low']) * 
                                 net_gradient_flow / (current['close'] + 1e-8))
        
        gradient_efficiency = (((positive_gradient - negative_gradient) / 
                              (positive_gradient + negative_gradient + 1e-8)) * 
                              pressure_efficiency)
        
        fractal_gradient_alignment = fractal_scaled_gradient * gradient_efficiency
        
        # Pressure Regime Framework
        pressure_acceleration = (micro_pressure > meso_pressure and 
                               meso_pressure > macro_pressure)
        pressure_deceleration = (micro_pressure < meso_pressure and 
                               meso_pressure < macro_pressure)
        pressure_transition = not (pressure_acceleration or pressure_deceleration)
        
        # Fractal-Specific Signals
        expanding_up_signal = (((current['close'] / prev_1['close']) - 1) * 
                              (current['amount'] / (current['high'] - current['low'] + 1e-8)))
        
        contracting_down_signal = (((current['close'] - current['open']) / 
                                  (current['open'] + 1e-8)) * 
                                  (current['volume'] / (current['amount'] + 1e-8)))
        
        mixed_fractal_signal = (((current['open'] - prev_1['close']) / 
                               (prev_1['close'] + 1e-8)) * 
                               (current['amount'] / (prev_1['amount'] + 1e-8)))
        
        # Gradient Regime Classification
        gradient_ratio = ((positive_gradient - negative_gradient) / 
                         (positive_gradient + negative_gradient + 1e-8))
        
        if i >= 1:
            pos_grad_prev = ((df.iloc[i-1]['close'] - df.iloc[i-1]['low']) / 
                           (df.iloc[i-1]['high'] - df.iloc[i-1]['low'] + 1e-8) * 
                           df.iloc[i-1]['amount'])
            neg_grad_prev = ((df.iloc[i-1]['high'] - df.iloc[i-1]['close']) / 
                           (df.iloc[i-1]['high'] - df.iloc[i-1]['low'] + 1e-8) * 
                           df.iloc[i-1]['amount'])
            
            sustained_positive = (gradient_ratio > 0 and 
                                positive_gradient > pos_grad_prev)
            sustained_negative = (gradient_ratio < 0 and 
                                negative_gradient > neg_grad_prev)
            gradient_reversal = not (sustained_positive or sustained_negative)
        else:
            sustained_positive = False
            sustained_negative = False
            gradient_reversal = True
        
        # Multi-Fractal Temporal Architecture
        # Ultra-Short Dynamics
        opening_gradient = ((current['close'] - current['open']) / 
                          (abs(current['open'] - prev_1['close']) + 1e-8))
        
        if i >= 1:
            pos_grad_prev = ((df.iloc[i-1]['close'] - df.iloc[i-1]['low']) / 
                           (df.iloc[i-1]['high'] - df.iloc[i-1]['low'] + 1e-8) * 
                           df.iloc[i-1]['amount'])
            neg_grad_prev = ((df.iloc[i-1]['high'] - df.iloc[i-1]['close']) / 
                           (df.iloc[i-1]['high'] - df.iloc[i-1]['low'] + 1e-8) * 
                           df.iloc[i-1]['amount'])
            net_grad_prev = pos_grad_prev - neg_grad_prev
            micro_gradient = net_gradient_flow / (net_grad_prev + 1e-8)
        else:
            micro_gradient = 0
        
        ultra_short_factor = opening_gradient * micro_gradient * pressure_efficiency
        
        # Short-Term Dynamics
        volume_gradient = current['volume'] / (prev_3['volume'] + 1e-8)
        
        if i >= 6:
            pos_grad_prev6 = ((df.iloc[i-6]['close'] - df.iloc[i-6]['low']) / 
                            (df.iloc[i-6]['high'] - df.iloc[i-6]['low'] + 1e-8) * 
                            df.iloc[i-6]['amount'])
            neg_grad_prev6 = ((df.iloc[i-6]['high'] - df.iloc[i-6]['close']) / 
                            (df.iloc[i-6]['high'] - df.iloc[i-6]['low'] + 1e-8) * 
                            df.iloc[i-6]['amount'])
            net_grad_prev6 = pos_grad_prev6 - neg_grad_prev6
            gradient_ratio_st = net_gradient_flow / (net_grad_prev6 + 1e-8)
        else:
            gradient_ratio_st = 0
        
        short_term_factor = volume_gradient * gradient_ratio_st * medium_gradient
        
        # Medium-Term Dynamics
        efficiency_gradient = pressure_efficiency * volume_accel
        
        high_12d = max(df.iloc[i-11:i+1]['high'])
        low_12d = min(df.iloc[i-11:i+1]['low'])
        range_gradient = ((current['high'] - current['low']) / 
                         (high_12d - low_12d + 1e-8))
        
        medium_term_factor = efficiency_gradient * range_gradient
        
        # Adaptive Pressure Synthesis
        # Pressure Core
        if pressure_acceleration:
            pressure_core = micro_pressure_int * meso_pressure_int
        elif pressure_deceleration:
            pressure_core = meso_pressure_int * macro_pressure_int
        else:
            pressure_core = ((current['high'] - current['low']) / 
                           (current['close'] + 1e-8) * 
                           abs(micro_pressure - meso_pressure))
        
        # Fractal Core
        if expanding_up:
            fractal_core = fractal_gradient_alignment * gradient_momentum
        elif contracting_down:
            fractal_core = pressure_efficiency * volume_accel
        else:
            fractal_core = fractal_scaled_gradient * medium_gradient
        
        # Gradient Core
        if sustained_positive:
            if i >= 1:
                pos_grad_prev = ((df.iloc[i-1]['close'] - df.iloc[i-1]['low']) / 
                               (df.iloc[i-1]['high'] - df.iloc[i-1]['low'] + 1e-8) * 
                               df.iloc[i-1]['amount'])
                gradient_core = gradient_ratio * (positive_gradient / (pos_grad_prev + 1e-8))
            else:
                gradient_core = gradient_ratio
        elif sustained_negative:
            if i >= 1:
                neg_grad_prev = ((df.iloc[i-1]['high'] - df.iloc[i-1]['close']) / 
                               (df.iloc[i-1]['high'] - df.iloc[i-1]['low'] + 1e-8) * 
                               df.iloc[i-1]['amount'])
                gradient_core = gradient_ratio * (negative_gradient / (neg_grad_prev + 1e-8))
            else:
                gradient_core = gradient_ratio
        else:
            gradient_core = ((1 - abs(gradient_ratio)) * 
                           (current['amount'] / (prev_1['amount'] + 1e-8)))
        
        # Signal Integration
        core_signal = (np.sign(current['close'] / prev_1['close'] - 1) * 
                      np.sign((current['close'] - current['open']) / (current['open'] + 1e-8)) * 
                      (current['amount'] / (prev_1['amount'] + 1e-8)))
        
        fractal_adjustment = core_signal / (current['high'] - current['low'] + 1e-8)
        
        enhanced_signal = (fractal_adjustment * 
                         np.sign((current['open'] - prev_1['close']) / (prev_1['close'] + 1e-8)))
        
        # Fractal Structure Enhancement
        range_exp_count = 0
        for j in range(max(0, i-3), i+1):
            if j >= 1:
                current_range = df.iloc[j]['high'] - df.iloc[j]['low']
                prev_range = df.iloc[j-1]['high'] - df.iloc[j-1]['low']
                if current_range > prev_range:
                    range_exp_count += 1
        
        fractal_structure = (((current['high'] - current['low']) / 
                            (prev_1['high'] - prev_1['low'] + 1e-8)) * 
                            range_exp_count)
        
        if expanding_up:
            fractal_alpha = expanding_up_signal
        elif contracting_down:
            fractal_alpha = contracting_down_signal
        else:
            fractal_alpha = mixed_fractal_signal
        
        structure_enhanced_alpha = fractal_alpha * enhanced_signal * fractal_structure
        
        # Multi-Scale Pressure Enhancement
        ultra_short_enhancement = pressure_core * ultra_short_factor
        short_term_enhancement = fractal_core * short_term_factor
        medium_term_enhancement = gradient_core * medium_term_factor
        
        # Final Alpha Construction
        pressure_gradient_synthesis = (ultra_short_enhancement + 
                                     short_term_enhancement + 
                                     medium_term_enhancement)
        
        fractal_adaptive_weighting = (pressure_gradient_synthesis * 
                                    structure_enhanced_alpha)
        
        final_alpha = (fractal_adaptive_weighting * 
                      (positive_gradient / (negative_gradient + 1e-8)))
        
        result.iloc[i] = final_alpha
    
    return result
