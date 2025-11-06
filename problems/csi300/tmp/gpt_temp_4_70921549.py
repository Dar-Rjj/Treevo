import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Helper functions
    def true_range(high, low, close_prev):
        return np.maximum(high - low, np.maximum(np.abs(high - close_prev), np.abs(low - close_prev)))
    
    def moving_average(series, window):
        return series.rolling(window=window, min_periods=1).mean()
    
    def rolling_sum(series, window):
        return series.rolling(window=window, min_periods=1).sum()
    
    def rolling_corr(series1, series2, window):
        return series1.rolling(window=window, min_periods=1).corr(series2)
    
    # Fractal Momentum Compression Framework
    data['ultra_short_momentum_comp'] = ((data['close'] - data['close'].shift(2)) / 
                                       (data['high'].shift(2) - data['low'].shift(2))) - \
                                      ((data['close'] - data['close'].shift(5)) / 
                                       (data['high'].shift(5) - data['low'].shift(5)))
    
    data['short_medium_momentum_comp'] = ((data['close'] - data['close'].shift(5)) / 
                                        (data['high'].shift(5) - data['low'].shift(5))) - \
                                       ((data['close'] - data['close'].shift(10)) / 
                                        (data['high'].shift(10) - data['low'].shift(10)))
    
    data['momentum_acceleration_comp'] = ((data['close'] - 2 * data['close'].shift(5) + data['close'].shift(10)) / 
                                        (data['high'].shift(10) - data['low'].shift(10)))
    
    # Volume-Weighted Momentum Enhancement
    data['close_ret'] = data['close'] - data['close'].shift(1)
    data['volume_weighted_momentum_comp'] = (rolling_sum(data['close_ret'] * data['volume'], 5) / 
                                           rolling_sum(data['volume'], 5))
    
    data['volume_momentum_alignment'] = np.sign(data['close'] - data['close'].shift(1)) * \
                                      np.sign(data['volume'] - data['volume'].shift(1))
    
    data['enhanced_momentum_comp'] = data['momentum_acceleration_comp'] * data['volume_momentum_alignment']
    
    # Fractal Momentum Regime Classification
    data['expanding_momentum'] = (data['ultra_short_momentum_comp'] > 0) & (data['short_medium_momentum_comp'] > 0)
    data['contracting_momentum'] = (data['ultra_short_momentum_comp'] < 0) & (data['short_medium_momentum_comp'] < 0)
    data['mixed_momentum'] = ~(data['expanding_momentum'] | data['contracting_momentum'])
    
    # Volume-Weighted Gap Efficiency Dynamics
    data['short_term_gap_eff'] = np.abs(data['close'] - data['open']) / rolling_sum(data['high'] - data['low'], 3)
    data['medium_term_gap_eff'] = np.abs(data['close'] - data['open']) / rolling_sum(data['high'] - data['low'], 10)
    data['gap_eff_momentum_comp'] = (data['medium_term_gap_eff'] - data['short_term_gap_eff']) * np.sign(data['close'] - data['open'])
    
    # Volume-Pressure Gap Analysis
    data['volume_pressure_asymmetry'] = (rolling_sum(data['volume'] * (data['close'] > data['open']), 5) - 
                                       rolling_sum(data['volume'] * (data['close'] < data['open']), 5)) / \
                                      rolling_sum(data['volume'], 5)
    
    data['gap_fill_pressure'] = (data['close'] - data['open']) / np.abs(data['open'] - data['close'].shift(1))
    data['volume_weighted_gap_eff'] = data['gap_eff_momentum_comp'] * data['volume_pressure_asymmetry']
    
    # Gap-Fractal Interaction
    data['fractal_gap_response'] = ((data['high'] - data['low']) / (data['high'].shift(5) - data['low'].shift(5))) * \
                                 np.abs(data['open'] / data['close'].shift(1) - 1)
    
    data['persistent_gap_signal'] = rolling_sum(np.sign(data['open'] - data['close'].shift(1)), 3) * data['gap_fill_pressure']
    data['enhanced_gap_eff'] = data['volume_weighted_gap_eff'] * data['fractal_gap_response']
    
    # Volatility-Volume Asymmetric Confirmation
    data['true_range'] = true_range(data['high'], data['low'], data['close'].shift(1))
    data['volatility_persistence'] = data['true_range'] / moving_average(data['true_range'], 5)
    
    data['multi_scale_volatility_asymmetry'] = ((data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1))) - \
                                             ((data['high'].shift(5) - data['low'].shift(5)) / 
                                              (data['high'].shift(6) - data['low'].shift(6)))
    
    data['volume_volatility_ratio'] = data['volume'] / data['true_range']
    data['efficient_volume'] = data['volume'] * np.abs(data['close'] - data['close'].shift(1)) / data['true_range']
    
    data['fractal_vol_vol_asymmetry'] = rolling_corr(data['volume'] / data['volume'].shift(1), 
                                                   (data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1)), 5)
    
    data['volatility_pressure_signal'] = data['volatility_persistence'] * data['multi_scale_volatility_asymmetry']
    data['volume_efficiency_signal'] = data['efficient_volume'] * data['fractal_vol_vol_asymmetry']
    data['asymmetric_confirmation'] = data['volatility_pressure_signal'] * data['volume_efficiency_signal']
    
    # Momentum-Gap Coherence Framework
    data['ultra_short_momentum'] = data['close'] / data['close'].shift(2) - 1
    data['short_medium_momentum'] = data['close'] / data['close'].shift(8) - 1
    data['long_term_momentum'] = data['close'] / data['close'].shift(21) - 1
    
    data['gap_momentum_convergence'] = data['ultra_short_momentum'] * data['short_medium_momentum'] * np.sign(data['close'] - data['open'])
    data['gap_momentum_divergence'] = np.abs(data['ultra_short_momentum'] - data['long_term_momentum']) * np.abs(data['open'] / data['close'].shift(1) - 1)
    data['coherence_strength'] = data['gap_momentum_convergence'] / (data['gap_momentum_divergence'] + 0.001)
    
    data['momentum_compression_coherence'] = data['volume_weighted_momentum_comp'] * data['momentum_acceleration_comp']
    data['gap_efficiency_coherence'] = data['enhanced_gap_eff'] * data['persistent_gap_signal']
    data['overall_coherence'] = data['momentum_compression_coherence'] * data['gap_efficiency_coherence'] * data['coherence_strength']
    
    # Adaptive Alpha Synthesis
    data['momentum_compression_signal'] = data['enhanced_momentum_comp'] * data['volume_weighted_momentum_comp']
    data['gap_efficiency_signal'] = data['enhanced_gap_eff'] * data['volume_pressure_asymmetry']
    data['volatility_confirmation_signal'] = data['asymmetric_confirmation'] * data['volume_volatility_ratio']
    
    # Regime-Adaptive Weighting
    data['expanding_momentum_weight'] = data['ultra_short_momentum_comp'] * data['volume'] / moving_average(data['volume'], 10)
    data['contracting_momentum_weight'] = data['short_medium_momentum_comp'] * moving_average(data['volume'], 5) / moving_average(data['volume'], 20)
    data['mixed_momentum_weight'] = data['momentum_acceleration_comp'] * data['volume'] / moving_average(data['volume'], 3) * \
                                  (data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1))
    
    # Final Alpha Output
    data['expanding_momentum_alpha'] = data['momentum_compression_signal'] * data['expanding_momentum_weight'] * data['gap_momentum_convergence']
    data['contracting_momentum_alpha'] = data['gap_efficiency_signal'] * data['contracting_momentum_weight'] * data['gap_momentum_divergence']
    data['mixed_momentum_alpha'] = data['volatility_confirmation_signal'] * data['mixed_momentum_weight'] * data['overall_coherence']
    
    data['composite_alpha'] = data['expanding_momentum_alpha'] + data['contracting_momentum_alpha'] + data['mixed_momentum_alpha']
    
    # Multi-Fractal Enhancement
    data['high_efficiency_regime'] = (data['gap_eff_momentum_comp'] > 0.1) & (data['volume_weighted_momentum_comp'] > 0)
    data['low_efficiency_regime'] = (data['gap_eff_momentum_comp'] < -0.1) & (data['volume_weighted_momentum_comp'] < 0)
    data['transition_regime'] = (np.sign((data['close'] - data['low']) / (data['high'] - data['low']) - 0.5) != 
                               np.sign((data['close'].shift(1) - data['low'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1)) - 0.5))
    
    # Regime-Specific Enhancement
    data['high_efficiency_enhancement'] = data['composite_alpha'] * (data['high'] - data['close']) / (data['close'] - data['low'])
    
    def movement_coherence_calc(close, open_price, volume, volume_prev):
        return np.sum(np.sign(close - open_price) == np.sign(volume - volume_prev)) / 3
    
    movement_coherence = []
    for i in range(len(data)):
        if i >= 2:
            window_data = data.iloc[i-2:i+1]
            coherence = movement_coherence_calc(window_data['close'], window_data['open'], 
                                              window_data['volume'], window_data['volume'].shift(1))
            movement_coherence.append(coherence)
        else:
            movement_coherence.append(0)
    data['movement_coherence'] = movement_coherence
    
    data['low_efficiency_enhancement'] = data['composite_alpha'] * data['movement_coherence']
    
    # Volume Cluster Asymmetry
    data['volume_cluster_asymmetry'] = data['volume'] / moving_average(data['volume'], 3) * \
                                     (data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1))
    
    data['fractal_cluster_asymmetry'] = data['volume_cluster_asymmetry'] * \
                                      rolling_corr(data['volume_cluster_asymmetry'], 
                                                 data['volume_cluster_asymmetry'].shift(1), 4)
    
    data['transition_enhancement'] = data['composite_alpha'] * data['fractal_cluster_asymmetry']
    
    # Final Enhanced Alpha
    data['high_efficiency_strength'] = np.abs(data['gap_eff_momentum_comp']) + np.abs(data['volume_weighted_momentum_comp'])
    data['low_efficiency_strength'] = np.abs(data['gap_eff_momentum_comp']) + np.abs(data['volume_weighted_momentum_comp'])
    data['transition_strength'] = np.abs(np.sign((data['close'] - data['low']) / (data['high'] - data['low']) - 0.5) - 
                                       np.sign((data['close'].shift(1) - data['low'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1)) - 0.5))
    
    total_strength = data['high_efficiency_strength'] + data['low_efficiency_strength'] + data['transition_strength']
    data['high_efficiency_weight'] = data['high_efficiency_strength'] / total_strength
    data['low_efficiency_weight'] = data['low_efficiency_strength'] / total_strength
    data['transition_weight'] = data['transition_strength'] / total_strength
    
    data['high_efficiency_component'] = data['high_efficiency_enhancement'] * data['high_efficiency_weight']
    data['low_efficiency_component'] = data['low_efficiency_enhancement'] * data['low_efficiency_weight']
    data['transition_component'] = data['transition_enhancement'] * data['transition_weight']
    
    data['final_enhanced_alpha'] = data['high_efficiency_component'] + data['low_efficiency_component'] + data['transition_component']
    
    return data['final_enhanced_alpha']
