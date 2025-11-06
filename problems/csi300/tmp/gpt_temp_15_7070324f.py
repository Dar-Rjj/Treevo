import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    for i in range(10, len(df)):
        current_data = df.iloc[:i+1].copy()
        
        # Fractal Gap Divergence Framework
        # Intraday Fractal Gap
        intraday_fractal_gap = (current_data['open'].iloc[-1] - current_data['low'].iloc[-1]) / \
                              (current_data['high'].iloc[-1] - current_data['low'].iloc[-1] + 1e-8)
        
        # Overnight Fractal Gap
        overnight_fractal_gap = (current_data['open'].iloc[-1] - current_data['close'].iloc[-2]) / \
                               (current_data['high'].iloc[-2] - current_data['low'].iloc[-2] + 1e-8)
        
        # Gap-Fill Efficiency
        gap_fill_efficiency = (current_data['close'].iloc[-1] - current_data['open'].iloc[-1]) / \
                             (current_data['open'].iloc[-1] - current_data['close'].iloc[-2] + 1e-8)
        
        # Gap Divergence Intensity
        volume_lead_indicator = current_data['volume'].iloc[-1] / (current_data['volume'].iloc[-5:-1].mean() + 1e-8)
        price_lead_indicator = current_data['close'].iloc[-1] / current_data['close'].iloc[-2] - 1
        
        bull_gap_divergence = intraday_fractal_gap * volume_lead_indicator
        bear_gap_divergence = overnight_fractal_gap * price_lead_indicator
        net_gap_divergence = bull_gap_divergence - bear_gap_divergence
        
        # Fractal Efficiency Divergence
        # 5-Day Fractal Efficiency
        high_low_sum_5 = (current_data['high'].iloc[-5:] - current_data['low'].iloc[-5:]).sum()
        fractal_efficiency_5 = abs(current_data['close'].iloc[-1] - current_data['close'].iloc[-5]) / (high_low_sum_5 + 1e-8)
        
        # 10-Day Fractal Efficiency
        high_low_sum_10 = (current_data['high'].iloc[-10:] - current_data['low'].iloc[-10:]).sum()
        fractal_efficiency_10 = abs(current_data['close'].iloc[-1] - current_data['close'].iloc[-10]) / (high_low_sum_10 + 1e-8)
        
        # Efficiency Momentum
        efficiency_momentum = fractal_efficiency_5 - fractal_efficiency_10
        
        # Fractal Range Efficiency
        fractal_range_efficiency = (current_data['high'].iloc[-1] - current_data['low'].iloc[-1]) / \
                                  (abs(current_data['close'].iloc[-1] - current_data['open'].iloc[-1]) + 1e-8)
        
        # Efficiency-Volume Divergence
        volume_efficiency_pressure = (current_data['close'].iloc[-1] - current_data['open'].iloc[-1]) / \
                                    (current_data['volume'].iloc[-1] + 1e-8) * efficiency_momentum
        
        amount_efficiency_divergence = current_data['volume'].iloc[-1] / (current_data['amount'].iloc[-1] + 1e-8) * volume_lead_indicator
        
        price_impact_divergence = abs(current_data['close'].iloc[-1] - current_data['open'].iloc[-1]) / \
                                 (current_data['amount'].iloc[-1] / (current_data['volume'].iloc[-1] + 1e-8) + 1e-8) * price_lead_indicator
        
        # Multi-Period Divergence Structure
        # Gap Divergence Persistence (3-day)
        gap_divergence_values = []
        for j in range(3):
            if i-j >= 10:
                intraday_gap_j = (current_data['open'].iloc[-1-j] - current_data['low'].iloc[-1-j]) / \
                                (current_data['high'].iloc[-1-j] - current_data['low'].iloc[-1-j] + 1e-8)
                overnight_gap_j = (current_data['open'].iloc[-1-j] - current_data['close'].iloc[-2-j]) / \
                                 (current_data['high'].iloc[-2-j] - current_data['low'].iloc[-2-j] + 1e-8)
                bull_div_j = intraday_gap_j * (current_data['volume'].iloc[-1-j] / (current_data['volume'].iloc[-5-j:-1-j].mean() + 1e-8))
                bear_div_j = overnight_gap_j * (current_data['close'].iloc[-1-j] / current_data['close'].iloc[-2-j] - 1)
                gap_divergence_values.append(bull_div_j - bear_div_j)
        
        gap_divergence_trend_3 = np.mean(gap_divergence_values) if gap_divergence_values else 0
        
        # Gap Divergence Acceleration
        gap_divergence_acceleration = net_gap_divergence - gap_divergence_values[1] if len(gap_divergence_values) > 1 else 0
        
        # Gap Divergence Consistency
        gap_divergence_consistency = net_gap_divergence * gap_divergence_values[1] if len(gap_divergence_values) > 1 else 0
        
        # Efficiency Divergence Core
        efficiency_divergence_core = efficiency_momentum * volume_efficiency_pressure
        
        # Multi-Scale Efficiency Alignment
        multi_scale_efficiency_alignment = np.sign(efficiency_momentum) * np.sign(volume_efficiency_pressure)
        
        # Efficiency Divergence Quality
        efficiency_divergence_quality = abs(efficiency_divergence_core) * amount_efficiency_divergence
        
        # Microstructure Regime Framework
        # Fractal Volatility Regime
        avg_high_low_5 = (current_data['high'].iloc[-5:] - current_data['low'].iloc[-5:]).mean()
        fractal_volatility_ratio = (current_data['high'].iloc[-1] - current_data['low'].iloc[-1]) / (avg_high_low_5 + 1e-8)
        
        # Range Ratio
        max_high_5 = current_data['high'].iloc[-5:].max()
        min_low_5 = current_data['low'].iloc[-5:].min()
        range_ratio = (current_data['high'].iloc[-1] - current_data['low'].iloc[-1]) / (max_high_5 - min_low_5 + 1e-8)
        
        # True Range and Volatility Regime
        tr1 = current_data['high'].iloc[-1] - current_data['low'].iloc[-1]
        tr2 = abs(current_data['high'].iloc[-1] - current_data['close'].iloc[-2])
        tr3 = abs(current_data['low'].iloc[-1] - current_data['close'].iloc[-2])
        true_range = max(tr1, tr2, tr3)
        
        avg_true_range_6 = np.mean([max(current_data['high'].iloc[-k] - current_data['low'].iloc[-k],
                                      abs(current_data['high'].iloc[-k] - current_data['close'].iloc[-k-1]),
                                      abs(current_data['low'].iloc[-k] - current_data['close'].iloc[-k-1]))
                                   for k in range(1, 7)])
        volatility_regime = true_range / (avg_true_range_6 + 1e-8)
        
        # Regime Multiplier
        if fractal_volatility_ratio > 1.5:
            regime_multiplier = 1.4
        elif fractal_volatility_ratio < 0.6:
            regime_multiplier = 0.6
        else:
            regime_multiplier = 1.0
        
        # Range-Position Regime
        # 3-Day Range Position
        min_low_3 = current_data['low'].iloc[-3:].min()
        max_high_3 = current_data['high'].iloc[-3:].max()
        range_position_3 = (current_data['close'].iloc[-1] - min_low_3) / (max_high_3 - min_low_3 + 1e-8)
        
        # 8-Day Range Position
        min_low_8 = current_data['low'].iloc[-8:].min()
        max_high_8 = current_data['high'].iloc[-8:].max()
        range_position_8 = (current_data['close'].iloc[-1] - min_low_8) / (max_high_8 - min_low_8 + 1e-8)
        
        # Range Convergence
        range_convergence = (range_position_3 - 0.5) * (range_position_8 - 0.5)
        
        # Intraday Microstructure Patterns
        opening_pressure = (current_data['open'].iloc[-1] - current_data['low'].iloc[-2]) / \
                          (current_data['high'].iloc[-2] - current_data['low'].iloc[-2] + 1e-8)
        
        closing_pressure = (current_data['close'].iloc[-1] - current_data['low'].iloc[-1]) / \
                          (current_data['high'].iloc[-1] - current_data['low'].iloc[-1] + 1e-8)
        
        intraday_shift = closing_pressure - opening_pressure
        
        # Volume-Amount Regime
        volume_concentration_fractal = current_data['volume'].iloc[-1] / \
                                      (current_data['amount'].iloc[-1] / ((current_data['high'].iloc[-1] + 
                                                                         current_data['low'].iloc[-1] + 
                                                                         current_data['close'].iloc[-1])/3) + 1e-8)
        
        amount_flow_transition = current_data['amount'].iloc[-1] / (current_data['amount'].iloc[-4] + 1e-8) - \
                                current_data['amount'].iloc[-4] / (current_data['amount'].iloc[-7] + 1e-8)
        
        liquidity_momentum = (current_data['volume'].iloc[-1] / (current_data['volume'].iloc[-6] + 1e-8) - 
                             current_data['volume'].iloc[-6] / (current_data['volume'].iloc[-11] + 1e-8)) * amount_flow_transition
        
        # Regime Alignment Metrics
        volatility_liquidity_alignment = volatility_regime * liquidity_momentum
        range_volume_alignment = range_convergence * volume_concentration_fractal
        intraday_regime_alignment = intraday_shift * opening_pressure * closing_pressure
        
        # Fractal Divergence-Regime Convergence
        # Gap-Efficiency Regime Integration
        volatility_weighted_gap_divergence = net_gap_divergence * regime_multiplier
        range_confirmed_gap_divergence = gap_divergence_trend_3 * range_convergence
        intraday_gap_alignment = gap_divergence_acceleration * intraday_shift
        
        regime_weighted_efficiency = efficiency_divergence_core * volatility_liquidity_alignment
        volume_efficiency_confirmation = efficiency_divergence_quality * volume_concentration_fractal
        multi_scale_efficiency_regime = multi_scale_efficiency_alignment * range_volume_alignment
        
        # Divergence-Structure Convergence
        structure_divergence_momentum = gap_divergence_consistency * intraday_regime_alignment
        range_position_divergence = range_convergence * price_impact_divergence
        
        timeframe_consistency = np.sign(intraday_shift) * np.sign(range_convergence) * abs(structure_divergence_momentum)
        
        # Convergence Quality Assessment
        signal_strength = abs(efficiency_divergence_core) * amount_efficiency_divergence
        
        # Regime Persistence
        if i > 11:
            volatility_alignment_prev = volatility_liquidity_alignment
            # Calculate previous volatility_liquidity_alignment
            tr_prev = max(current_data['high'].iloc[-2] - current_data['low'].iloc[-2],
                         abs(current_data['high'].iloc[-2] - current_data['close'].iloc[-3]),
                         abs(current_data['low'].iloc[-2] - current_data['close'].iloc[-3]))
            avg_tr_prev = np.mean([max(current_data['high'].iloc[-k-1] - current_data['low'].iloc[-k-1],
                                     abs(current_data['high'].iloc[-k-1] - current_data['close'].iloc[-k-2]),
                                     abs(current_data['low'].iloc[-k-1] - current_data['close'].iloc[-k-2]))
                                 for k in range(1, 7)])
            volatility_regime_prev = tr_prev / (avg_tr_prev + 1e-8)
            
            amount_flow_prev = current_data['amount'].iloc[-2] / (current_data['amount'].iloc[-5] + 1e-8) - \
                              current_data['amount'].iloc[-5] / (current_data['amount'].iloc[-8] + 1e-8)
            
            liquidity_momentum_prev = (current_data['volume'].iloc[-2] / (current_data['volume'].iloc[-7] + 1e-8) - 
                                      current_data['volume'].iloc[-7] / (current_data['volume'].iloc[-12] + 1e-8)) * amount_flow_prev
            
            volatility_alignment_prev = volatility_regime_prev * liquidity_momentum_prev
            regime_persistence = np.sign(volatility_liquidity_alignment) * np.sign(volatility_alignment_prev) * abs(volatility_liquidity_alignment)
        else:
            regime_persistence = 0
        
        divergence_quality = signal_strength * regime_persistence * timeframe_consistency
        
        # Core Convergence Synthesis
        regime_adapted_gap_efficiency = volatility_weighted_gap_divergence * regime_weighted_efficiency
        volume_confirmed_divergence = range_confirmed_gap_divergence * volume_efficiency_confirmation
        structure_aligned_momentum = intraday_gap_alignment * structure_divergence_momentum
        
        quality_enhanced_convergence = regime_adapted_gap_efficiency * divergence_quality
        regime_structure_alignment = volume_confirmed_divergence * multi_scale_efficiency_regime
        multi_factor_confirmation = structure_aligned_momentum * range_position_divergence
        
        # Composite Fractal Divergence Alpha
        core_alpha_factor = quality_enhanced_convergence * regime_structure_alignment
        microstructure_refinement = core_alpha_factor * multi_factor_confirmation
        final_factor = microstructure_refinement * liquidity_momentum
        
        result.iloc[i] = final_factor
    
    # Apply 3-day simple moving average smoothing
    result_smoothed = result.rolling(window=3, min_periods=1).mean()
    
    return result_smoothed
