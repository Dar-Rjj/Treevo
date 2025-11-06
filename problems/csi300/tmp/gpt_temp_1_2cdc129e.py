import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Extract price and volume data
    open_p = df['open']
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']
    
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    # Calculate required windows
    for i in range(19, len(df)):
        if i < 19:
            continue
            
        # Current day data
        current_data = {
            'open': open_p.iloc[i],
            'high': high.iloc[i], 
            'low': low.iloc[i],
            'close': close.iloc[i],
            'volume': volume.iloc[i]
        }
        
        # Historical data
        hist_data = {
            'close_1': close.iloc[i-1], 'close_2': close.iloc[i-2], 'close_3': close.iloc[i-3],
            'close_4': close.iloc[i-4], 'close_5': close.iloc[i-5], 'close_10': close.iloc[i-10],
            'close_19': close.iloc[i-19],
            'high_1': high.iloc[i-1], 'high_2': high.iloc[i-2], 'high_5': high.iloc[i-5],
            'low_1': low.iloc[i-1], 'low_2': low.iloc[i-2], 'low_5': low.iloc[i-5],
            'volume_1': volume.iloc[i-1], 'volume_2': volume.iloc[i-2], 'volume_5': volume.iloc[i-5],
            'volume_10': volume.iloc[i-10], 'volume_9': volume.iloc[i-9]
        }
        
        # Multi-Scale Pressure-Decay Components
        # Short-term Pressure-Decay
        if hist_data['high_2'] != hist_data['low_2'] and current_data['high'] != current_data['low']:
            short_term_pressure = (
                (current_data['close'] - hist_data['close_2']) / (hist_data['high_2'] - hist_data['low_2']) *
                (1 - current_data['volume'] / hist_data['volume_2']) *
                (current_data['close'] - current_data['open']) / (current_data['high'] - current_data['low'])
            )
        else:
            short_term_pressure = 0
            
        # Medium-term Pressure-Decay
        if hist_data['high_5'] != hist_data['low_5'] and current_data['high'] != current_data['low']:
            medium_term_pressure = (
                (current_data['close'] - hist_data['close_5']) / (hist_data['high_5'] - hist_data['low_5']) *
                (1 - current_data['volume'] / hist_data['volume_5']) *
                (current_data['close'] - current_data['open']) / (current_data['high'] - current_data['low'])
            )
        else:
            medium_term_pressure = 0
            
        # Pressure-Decay Acceleration
        pressure_decay_acceleration = short_term_pressure - medium_term_pressure
        
        # Pressure Return Convergence Framework
        # Pressure Return Component
        if hist_data['close_4'] != 0 and hist_data['close_19'] != 0 and current_data['high'] != current_data['low']:
            pressure_return_component = (
                (current_data['close'] / hist_data['close_4'] - 1) *
                (current_data['close'] / hist_data['close_19'] - 1) *
                (current_data['close'] - current_data['open']) / (current_data['high'] - current_data['low'])
            )
        else:
            pressure_return_component = 0
            
        # Gap-Pressure Integration
        if hist_data['high_1'] != hist_data['low_1'] and current_data['high'] != current_data['low']:
            gap_pressure_integration = (
                (current_data['open'] - hist_data['close_1']) / (hist_data['high_1'] - hist_data['low_1']) *
                (current_data['close'] - current_data['open']) / (current_data['high'] - current_data['low'])
            )
        else:
            gap_pressure_integration = 0
            
        # Pressure Convergence Factor
        pressure_convergence_factor = pressure_return_component * gap_pressure_integration
        
        # Volume-Compression Validation System
        # Volume per Unit Pressure
        if current_data['high'] != current_data['low']:
            volume_per_unit_pressure = current_data['volume'] / (current_data['high'] - current_data['low'])
        else:
            volume_per_unit_pressure = 0
            
        # Pressure Volume Spike
        volume_window_4 = volume.iloc[i-4:i+1]
        if len(volume_window_4) > 0 and volume_window_4.mean() != 0:
            pressure_volume_spike = current_data['volume'] / volume_window_4.mean()
        else:
            pressure_volume_spike = 0
            
        # Volume-Pressure Correlation
        pressure_ratios = []
        volume_window_corr = []
        for j in range(max(i-4, 0), i+1):
            if j >= 0 and high.iloc[j] != low.iloc[j]:
                pressure_ratio = (close.iloc[j] - open_p.iloc[j]) / (high.iloc[j] - low.iloc[j])
                pressure_ratios.append(pressure_ratio)
                volume_window_corr.append(volume.iloc[j])
        
        if len(pressure_ratios) >= 2:
            volume_pressure_correlation = np.corrcoef(volume_window_corr, pressure_ratios)[0, 1]
            if np.isnan(volume_pressure_correlation):
                volume_pressure_correlation = 0
        else:
            volume_pressure_correlation = 0
            
        # Compression-Pressure Integration
        # High-Low Compression
        if hist_data['high_1'] != hist_data['low_1']:
            high_low_compression = (current_data['high'] - current_data['low']) / (hist_data['high_1'] - hist_data['low_1'])
        else:
            high_low_compression = 1
            
        # Volume Compression Component
        if hist_data['volume_1'] != 0:
            volume_compression_component = current_data['volume'] / hist_data['volume_1']
        else:
            volume_compression_component = 1
            
        # Pressure-Compression Alignment
        if current_data['high'] != current_data['low']:
            pressure_compression_alignment = (
                (current_data['close'] - current_data['open']) / (current_data['high'] - current_data['low']) *
                high_low_compression
            )
        else:
            pressure_compression_alignment = 0
            
        # Breakout Quality Assessment Framework
        # Pressure Efficiency Volatility
        pressure_efficiency_window = []
        for j in range(max(i-4, 0), i+1):
            if j >= 0 and high.iloc[j] != low.iloc[j]:
                efficiency = (close.iloc[j] - open_p.iloc[j]) / (high.iloc[j] - low.iloc[j])
                pressure_efficiency_window.append(efficiency)
        
        if len(pressure_efficiency_window) >= 2:
            pressure_efficiency_volatility = 1 / (np.std(pressure_efficiency_window) + 1e-8)
        else:
            pressure_efficiency_volatility = 1
            
        # Positive Pressure Ratio
        positive_count = 0
        for j in range(max(i-4, 0), i+1):
            if j >= 5 and close.iloc[j] > close.iloc[j-5]:
                positive_count += 1
        positive_pressure_ratio = positive_count / 5 if i >= 9 else 0.5
        
        # Efficiency Consistency
        efficiency_consistency = positive_pressure_ratio * pressure_efficiency_volatility
        
        # Volume-Confirmed Breakout Signals
        # Volume Acceleration Component
        if i >= 10:
            volume_acceleration_component = (
                (current_data['volume'] - hist_data['volume_5']) - 
                (hist_data['volume_5'] - hist_data['volume_10'])
            )
        else:
            volume_acceleration_component = 0
            
        # Pressure-Volume Breakout
        pressure_volume_breakout = volume_pressure_correlation * pressure_volume_spike
        
        # Breakout Quality Score
        breakout_quality_score = efficiency_consistency * pressure_volume_breakout
        
        # Multi-Timeframe Coherence Validation
        # 3-day Pressure-Decay Trend
        if (i >= 3 and hist_data['high_2'] != hist_data['low_2'] and 
            current_data['high'] != current_data['low']):
            sign_close_2 = np.sign(current_data['close'] - hist_data['close_2'])
            sign_close_3 = np.sign(hist_data['close_1'] - hist_data['close_3'])
            three_day_trend = (
                sign_close_2 * sign_close_3 *
                (1 - current_data['volume'] / hist_data['volume_2']) *
                (current_data['close'] - current_data['open']) / (current_data['high'] - current_data['low'])
            )
        else:
            three_day_trend = 0
            
        # Acceleration Consistency
        if (i >= 1 and current_data['high'] != current_data['low'] and 
            hist_data['high_1'] != hist_data['low_1']):
            current_pressure = (current_data['close'] - current_data['open']) / (current_data['high'] - current_data['low'])
            prev_pressure = (hist_data['close_1'] - open_p.iloc[i-1]) / (hist_data['high_1'] - hist_data['low_1'])
            acceleration_consistency = current_pressure * prev_pressure * volume_acceleration_component
        else:
            acceleration_consistency = 0
            
        # Medium-term Regime Validation
        # Volume Efficiency Regime
        volume_efficiency_current = volume_per_unit_pressure
        volume_efficiency_window = []
        for j in range(max(i-4, 0), i+1):
            if j >= 0 and high.iloc[j] != low.iloc[j]:
                eff = volume.iloc[j] / (high.iloc[j] - low.iloc[j])
                volume_efficiency_window.append(eff)
        
        if len(volume_efficiency_window) > 0 and np.mean(volume_efficiency_window) != 0:
            volume_efficiency_regime = volume_efficiency_current / np.mean(volume_efficiency_window)
        else:
            volume_efficiency_regime = 1
            
        # Efficiency Decay Stability
        volume_efficiency_long_window = []
        for j in range(max(i-9, 0), i+1):
            if j >= 0 and high.iloc[j] != low.iloc[j]:
                eff = volume.iloc[j] / (high.iloc[j] - low.iloc[j])
                volume_efficiency_long_window.append(eff)
        
        if len(volume_efficiency_long_window) >= 2:
            efficiency_decay_stability = 1 / (np.std(volume_efficiency_long_window) + 1e-8)
        else:
            efficiency_decay_stability = 1
            
        # Regime Strength
        if hist_data['volume_9'] != 0:
            regime_strength = (
                volume_efficiency_regime * efficiency_decay_stability * 
                (1 - current_data['volume'] / hist_data['volume_9'])
            )
        else:
            regime_strength = volume_efficiency_regime * efficiency_decay_stability
            
        # Composite Pressure-Decay Alpha Generation
        # Core Pressure-Decay Factor
        core_pressure_decay_factor = pressure_decay_acceleration * pressure_convergence_factor
        
        # Volume-Validation Enhanced Factor
        volume_validation_enhanced_factor = (
            core_pressure_decay_factor * pressure_volume_breakout * volume_compression_component
        )
        
        # Breakout Quality Multiplier
        breakout_quality_multiplier = breakout_quality_score * pressure_compression_alignment
        
        # Coherence Validation Factor
        coherence_validation_factor = three_day_trend * regime_strength
        
        # Final Alpha Factor
        final_alpha = (
            volume_validation_enhanced_factor * 
            breakout_quality_multiplier * 
            coherence_validation_factor
        )
        
        alpha.iloc[i] = final_alpha
    
    # Fill NaN values with 0
    alpha = alpha.fillna(0)
    
    return alpha
