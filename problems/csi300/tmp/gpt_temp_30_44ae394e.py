import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Helper functions
    def true_range(high, low, close_prev):
        return np.maximum(high - low, 
                        np.maximum(np.abs(high - close_prev), 
                                  np.abs(low - close_prev)))
    
    # Multi-Scale Volatility & Temporal Momentum Analysis
    # Fractal Volatility-Timing Detection
    data['TR'] = true_range(data['high'], data['low'], data['close'].shift(1))
    data['TR_momentum'] = data['TR'] * (data['close'] - data['close'].shift(1))
    
    # Volatility-Timing Divergence
    data['vol_timing_div'] = (data['TR'] / data['TR'].shift(5)) - \
                            ((data['close'] - data['close'].shift(1)) / 
                             (data['close'].shift(5) - data['close'].shift(6)))
    
    # Temporal Regime Classification
    tr_median = data['TR'].rolling(window=20, min_periods=1).median()
    data['vol_regime'] = np.where(data['TR'] > 1.2 * tr_median, 1.2,
                                 np.where(data['TR'] < 0.8 * tr_median, 0.8, 1.0))
    data['temporal_regime'] = data['vol_regime'] * (data['volume'] / data['volume'].shift(1))
    
    # Structural Break & Gap Momentum
    data['price_level_momentum'] = (np.abs(data['close'] - (data['high'] + data['low'])/2) / 
                                   (data['high'] - data['low'])) * (data['close'] - data['close'].shift(1))
    
    data['gap_momentum'] = (np.abs(data['open'] - data['close'].shift(1)) / 
                           np.abs(data['close'].shift(1) - data['close'].shift(2))) * \
                          (data['volume'] / data['amount']) * np.sign(data['close'] - data['open'])
    
    data['break_momentum_intensity'] = data['price_level_momentum'] * data['gap_momentum']
    
    # Volatility-Momentum Convergence
    data['vol_break_momentum'] = data['TR'] * data['break_momentum_intensity']
    
    data['gap_range_alignment'] = ((data['close'] - data['open']) / (data['high'] - data['low']) - 
                                  (data['close'].shift(5) - data['open'].shift(5)) / 
                                  (data['high'].shift(5) - data['low'].shift(5))) * data['volume']
    
    data['temporal_regime_adaptive_break'] = data['vol_break_momentum'] * data['gap_range_alignment'] * data['vol_regime']
    
    # Hierarchical Pressure-Momentum Efficiency
    # Fractal Momentum Pressure
    data['price_efficiency_pressure'] = ((data['close'] - data['open']) / data['TR']) * \
                                       ((data['high']/data['high'].shift(1) - 1) - 
                                        (data['low']/data['low'].shift(1) - 1)) * data['volume']
    
    data['volume_timing_momentum'] = ((data['close'] - data['close'].shift(3)) / data['close'].shift(3) * data['volume'] - 
                                     (data['close'].shift(5) - data['close'].shift(8)) / data['close'].shift(8) * 
                                     data['volume'].shift(5)) * (data['volume'] / data['volume'].shift(1))
    
    data['efficiency_pressure_div'] = data['price_efficiency_pressure'] - data['volume_timing_momentum']
    
    # Microstructure Asymmetry Acceleration
    data['order_flow_asymmetry'] = ((data['close'] - data['low']) / (data['high'] - data['low']) - 0.5 - 
                                   ((data['close'].shift(5) - data['low'].shift(5)) / 
                                    (data['high'].shift(5) - data['low'].shift(5)) - 0.5)) * data['volume']
    
    # Volume Pressure Timing
    def volume_pressure_timing(data, window=5):
        result = []
        for i in range(len(data)):
            if i < window:
                result.append(0)
                continue
            current_window = data.iloc[i-window+1:i+1]
            prev_window = data.iloc[i-window*2+1:i-window+1]
            
            current_pressure = ((current_window[current_window['close'] > current_window['open']]['volume'].sum() - 
                               current_window[current_window['close'] < current_window['open']]['volume'].sum()) / 
                               current_window['volume'].sum())
            
            prev_pressure = ((prev_window[prev_window['close'] > prev_window['open']]['volume'].sum() - 
                            prev_window[prev_window['close'] < prev_window['open']]['volume'].sum()) / 
                            prev_window['volume'].sum())
            
            result.append((current_pressure - prev_pressure) * (data.iloc[i]['volume'] / data.iloc[i-1]['volume']))
        return pd.Series(result, index=data.index)
    
    data['volume_pressure_timing'] = volume_pressure_timing(data)
    data['pressure_timing_signal'] = data['order_flow_asymmetry'] * data['volume_pressure_timing']
    
    # Hierarchical Temporal Pressure
    data['primary_efficiency_layer'] = data['efficiency_pressure_div'] * data['pressure_timing_signal']
    data['secondary_pressure_amp'] = data['primary_efficiency_layer'] * data['volume_timing_momentum']
    data['hierarchical_pressure_cascade'] = data['primary_efficiency_layer'] * data['secondary_pressure_amp']
    
    # Compression-Breakout Temporal Integration
    # Volatility Compression Momentum
    data['range_compression_timing'] = ((data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1)) - 
                                       (data['high'].shift(5) - data['low'].shift(5)) / 
                                       (data['high'].shift(6) - data['low'].shift(6))) * data['volume']
    
    rolling_max_high = data['high'].rolling(window=5, min_periods=1).max()
    rolling_max_high_prev = data['high'].shift(5).rolling(window=5, min_periods=1).max()
    
    data['compression_breakout_momentum'] = ((data['close'] - rolling_max_high) * data['range_compression_timing'] - 
                                            (data['close'].shift(5) - rolling_max_high_prev) * 
                                            data['range_compression_timing'].shift(5))
    
    volume_avg_5 = data['volume'].rolling(window=5, min_periods=1).mean()
    data['volume_breakout_timing'] = data['compression_breakout_momentum'] * (data['volume'] / volume_avg_5.shift(1))
    
    # Gap-Compression Temporal Momentum
    data['gap_magnitude_timing'] = ((np.abs(data['open'] - data['close'].shift(1)) / 
                                   np.abs(data['close'].shift(1) - data['close'].shift(2)) - 
                                   np.abs(data['open'].shift(5) - data['close'].shift(6)) / 
                                   np.abs(data['close'].shift(6) - data['close'].shift(7)))) * data['volume']
    
    data['gap_compression_temporal'] = data['gap_magnitude_timing'] * data['range_compression_timing']
    data['temporal_enhanced_breakout'] = data['volume_breakout_timing'] * data['gap_compression_temporal']
    
    # Adaptive Temporal-Pressure Integration
    # Break-Enhanced Temporal Efficiency
    data['structural_break_temporal'] = data['efficiency_pressure_div'] * (1 / (data['break_momentum_intensity'] + 1e-8))
    data['pressure_transition_timing'] = data['pressure_timing_signal'] * data['gap_momentum']
    data['break_temporal_composite'] = data['structural_break_temporal'] * data['pressure_transition_timing']
    
    # Temporal Regime-Adaptive Pressure
    data['vol_timing_scaling'] = data['hierarchical_pressure_cascade'] * data['vol_regime'] * (data['volume'] / data['volume'].shift(1))
    data['regime_break_temporal'] = data['vol_timing_scaling'] * data['temporal_regime_adaptive_break']
    data['adaptive_temporal_framework'] = data['vol_timing_scaling'] * data['regime_break_temporal']
    
    # Integrated Temporal-Pressure System
    data['efficiency_temporal_fusion'] = data['break_temporal_composite'] * data['adaptive_temporal_framework']
    data['hierarchical_temporal_integration'] = data['efficiency_temporal_fusion'] * data['hierarchical_pressure_cascade']
    data['adaptive_temporal_network'] = data['efficiency_temporal_fusion'] * data['hierarchical_temporal_integration']
    
    # Temporal Asymmetry Signal Generation
    # Momentum-Timing Signals
    data['short_term_pressure'] = ((data['close'] - data['close'].shift(1)) / 
                                  (data['close'].shift(1) - data['close'].shift(2) + 1e-8)) * \
                                 data['volume'] * data['pressure_timing_signal']
    
    data['medium_term_efficiency'] = ((data['close'] - data['close'].shift(5)) / 
                                     (data['close'].shift(5) - data['close'].shift(10) + 1e-8)) * \
                                    (data['volume'] / volume_avg_5) * data['efficiency_pressure_div']
    
    data['temporal_regime_strength'] = data['temporal_regime'] * data['volume'] * (data['close'] - data['close'].shift(1))
    
    # Range-Timing Efficiency
    data['high_efficiency_temporal'] = ((data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)) * data['volume']
    data['low_efficiency_temporal'] = data['high_efficiency_temporal']
    data['range_expansion_timing'] = data['range_compression_timing'] * (data['volume'] / volume_avg_5)
    
    # Volume-Timing Asymmetry
    data['early_volume_pressure'] = data['volume_pressure_timing'] * (data['close'] - data['open'])
    data['volume_acceleration_timing'] = (data['volume'] / data['volume'].shift(1)) * \
                                        (data['volume'].shift(1) / data['volume'].shift(2)) * \
                                        (data['close'] - data['close'].shift(1)) * data['pressure_timing_signal']
    
    volume_avg_10 = data['volume'].rolling(window=10, min_periods=1).mean()
    data['volume_regime_timing'] = (data['volume'] / volume_avg_10) * \
                                  (data['volume'].shift(1) / volume_avg_10.shift(1)) * \
                                  (data['close'] - data['close'].shift(1)) * data['temporal_regime']
    
    # Fractal Temporal Asymmetry Alpha
    # Cross-Temporal Signal Synthesis
    data['temporal_pressure_synthesis'] = data['temporal_regime_adaptive_break'] * data['hierarchical_pressure_cascade']
    data['break_temporal_fusion'] = data['break_momentum_intensity'] * data['temporal_enhanced_breakout']
    data['fractal_temporal_network'] = data['temporal_pressure_synthesis'] * data['break_temporal_fusion']
    
    # Final Adaptive Temporal Factor
    momentum_signals = (data['short_term_pressure'] + data['medium_term_efficiency']) * data['temporal_regime_strength']
    range_efficiency = data['high_efficiency_temporal'] * data['vol_timing_scaling']
    volume_asymmetry = data['volume_acceleration_timing'] * data['volume_regime_timing']
    
    data['regime_weighted_temporal'] = momentum_signals + range_efficiency + volume_asymmetry
    data['fractal_temporal_convergence'] = data['regime_weighted_temporal'] * data['adaptive_temporal_network'] * (1 + data['range_expansion_timing'])
    
    # Final alpha factor
    alpha = data['fractal_temporal_convergence'].fillna(0)
    
    return alpha
