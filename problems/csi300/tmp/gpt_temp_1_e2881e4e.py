import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize result series
    result = pd.Series(index=data.index, dtype=float)
    
    # Calculate required lookback periods
    for i in range(len(data)):
        if i < 21:  # Need at least 21 days for long-term calculations
            result.iloc[i] = 0
            continue
            
        current_data = data.iloc[:i+1]  # Only use data up to current day
        
        # Extract current values
        close_t = current_data['close'].iloc[-1]
        high_t = current_data['high'].iloc[-1]
        low_t = current_data['low'].iloc[-1]
        volume_t = current_data['volume'].iloc[-1]
        
        # Multi-Timeframe Fractal Momentum Components
        # Short-term (3-day)
        close_t_minus_3 = current_data['close'].iloc[-4]
        st_returns = np.abs(current_data['close'].iloc[-3:-1].values - current_data['close'].iloc[-4:-2].values)
        st_fractal_momentum = (close_t - close_t_minus_3) / (np.sum(st_returns) + 1e-8)
        
        # Medium-term (8-day)
        close_t_minus_8 = current_data['close'].iloc[-9]
        mt_returns = np.abs(current_data['close'].iloc[-8:-1].values - current_data['close'].iloc[-9:-2].values)
        mt_fractal_momentum = (close_t - close_t_minus_8) / (np.sum(mt_returns) + 1e-8)
        
        # Long-term (21-day)
        close_t_minus_21 = current_data['close'].iloc[-22]
        lt_returns = np.abs(current_data['close'].iloc[-21:-1].values - current_data['close'].iloc[-22:-2].values)
        lt_fractal_momentum = (close_t - close_t_minus_21) / (np.sum(lt_returns) + 1e-8)
        
        # Volume-Weighted Fractal Momentum
        # 3-Day Volume-Momentum
        st_volume_momentum = 0
        st_vol_denominator = 0
        for j in range(1, 4):
            close_diff = current_data['close'].iloc[-j] - current_data['close'].iloc[-j-1]
            st_volume_momentum += current_data['volume'].iloc[-j] * close_diff
            st_vol_denominator += np.abs(close_diff)
        st_volume_momentum = st_volume_momentum / (st_vol_denominator + 1e-8)
        
        # 8-Day Volume-Momentum
        mt_volume_momentum = 0
        mt_vol_denominator = 0
        for j in range(1, 9):
            close_diff = current_data['close'].iloc[-j] - current_data['close'].iloc[-j-1]
            mt_volume_momentum += current_data['volume'].iloc[-j] * close_diff
            mt_vol_denominator += np.abs(close_diff)
        mt_volume_momentum = mt_volume_momentum / (mt_vol_denominator + 1e-8)
        
        # Fractal-Volume Divergence
        fractal_volume_divergence = np.sign(st_fractal_momentum) * np.sign(st_volume_momentum)
        
        # Range-Based Efficiency Assessment
        daily_range_efficiency = (close_t - low_t) / ((high_t - low_t) + 1e-8)
        
        # Multi-Day Range Utilization
        close_diff_3 = np.abs(close_t - current_data['close'].iloc[-4])
        high_max_3 = np.max(current_data['high'].iloc[-3:-1])
        low_min_3 = np.min(current_data['low'].iloc[-3:-1])
        multi_day_range_util = close_diff_3 / ((high_max_3 - low_min_3) + 1e-8)
        
        # Range Expansion Factor
        current_range = high_t - low_t
        prev_ranges = [current_data['high'].iloc[-j] - current_data['low'].iloc[-j] for j in range(2, 6)]
        range_expansion = current_range / ((np.mean(prev_ranges)) + 1e-8)
        
        # Volume-Pressure Dynamics System
        # Volume-Adjusted Return
        close_t_minus_5 = current_data['close'].iloc[-6]
        volume_t_minus_5 = current_data['volume'].iloc[-6]
        volume_adjusted_return = ((close_t / close_t_minus_5) - 1) * np.log((volume_t / volume_t_minus_5) + 1e-8)
        
        # Volume Trend Alignment
        volume_trend_alignment = np.sign(close_t - current_data['close'].iloc[-2]) * np.sign(volume_t - current_data['volume'].iloc[-2])
        
        # Volume Acceleration
        volume_acceleration = (volume_t / (current_data['volume'].iloc[-4] + 1e-8)) - (current_data['volume'].iloc[-4] / (current_data['volume'].iloc[-7] + 1e-8))
        
        # Pressure Dynamics Components
        buying_pressure_ratio = (close_t - low_t) / ((high_t - low_t) + 1e-8)
        
        pressure_volume_alignment = buying_pressure_ratio * (volume_t / (current_data['volume'].iloc[-2] + 1e-8))
        
        pressure_persistence = np.mean([(current_data['close'].iloc[-j] - current_data['low'].iloc[-j]) / 
                                      ((current_data['high'].iloc[-j] - current_data['low'].iloc[-j]) + 1e-8) 
                                      for j in range(1, 4)])
        
        # Volume-Pressure Divergence Detection
        volume_breakout_signal = (volume_t > np.mean([current_data['volume'].iloc[-j] for j in range(2, 5)])) * np.sign(close_t - current_data['close'].iloc[-2])
        
        pressure_momentum_divergence = np.sign(pressure_persistence) * np.sign(volume_adjusted_return)
        
        volume_weighted_pressure = buying_pressure_ratio * np.log(volume_t + 1)
        
        # Multi-Dimensional Divergence Analysis
        core_divergence = fractal_volume_divergence * volume_trend_alignment
        
        multi_timeframe_consistency = np.sign(st_fractal_momentum) * np.sign(mt_fractal_momentum) * np.sign(lt_fractal_momentum)
        
        divergence_momentum_alignment = np.sign(core_divergence) * np.sign(multi_timeframe_consistency)
        
        # Pressure-Enhanced Divergence
        pressure_weighted_divergence = core_divergence * pressure_persistence
        
        volume_confirmation_strength = volume_trend_alignment * volume_breakout_signal
        
        enhanced_pressure_divergence = pressure_weighted_divergence * volume_confirmation_strength
        
        # Range Utilization Integration
        range_adjusted_efficiency = daily_range_efficiency * range_expansion
        
        utilization_divergence_alignment = range_adjusted_efficiency * enhanced_pressure_divergence
        
        multi_timeframe_range_consistency = multi_day_range_util * range_expansion
        
        # Regime-Adaptive Combination Framework
        # Fractal Efficiency Regime Detection
        if mt_fractal_momentum > 0.6:
            regime = 'high'
            fractal_weight = 0.7
            volume_pressure_weight = 0.3
        elif mt_fractal_momentum >= 0.4:
            regime = 'medium'
            fractal_weight = 0.5
            volume_pressure_weight = 0.5
        else:
            regime = 'low'
            fractal_weight = 0.4
            volume_pressure_weight = 0.6
        
        # Breakout Confirmation Enhancement
        # Multi-Timeframe Breakout Detection
        short_term_breakout = int(close_t > np.max(current_data['high'].iloc[-3:-1]) or close_t < np.min(current_data['low'].iloc[-3:-1]))
        medium_term_breakout = int(close_t > np.max(current_data['high'].iloc[-8:-1]) or close_t < np.min(current_data['low'].iloc[-8:-1]))
        long_term_breakout = int(close_t > np.max(current_data['high'].iloc[-21:-1]) or close_t < np.min(current_data['low'].iloc[-21:-1]))
        
        breakout_count = short_term_breakout + medium_term_breakout + long_term_breakout
        
        breakout_momentum_alignment = np.sign(breakout_count) * np.sign(st_fractal_momentum)
        
        breakout_confidence = breakout_count * breakout_momentum_alignment
        
        # Pressure Confirmation
        pressure_direction = np.sign(buying_pressure_ratio - 0.5)
        pressure_momentum_alignment = pressure_direction * np.sign(st_fractal_momentum)
        pressure_multiplier = 1 + pressure_momentum_alignment
        
        # Final Alpha Construction
        # Primary Divergence Components
        core_divergence_factor = core_divergence * multi_timeframe_consistency
        volume_pressure_multiplier = enhanced_pressure_divergence * pressure_momentum_divergence
        range_utilization_modifier = range_adjusted_efficiency * multi_timeframe_range_consistency
        
        # Regime-Adaptive Signal Processing
        fractal_component = (core_divergence_factor + range_utilization_modifier) / 2
        volume_pressure_component = volume_pressure_multiplier
        
        primary_factor = (fractal_weight * fractal_component + 
                         volume_pressure_weight * volume_pressure_component)
        
        # Apply enhancements
        primary_factor_with_breakout = primary_factor * (1 + 0.1 * breakout_confidence)
        final_alpha = primary_factor_with_breakout * pressure_multiplier
        
        result.iloc[i] = final_alpha
    
    return result
