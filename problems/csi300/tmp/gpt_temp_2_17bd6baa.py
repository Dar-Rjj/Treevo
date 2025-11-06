import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Helper function for safe division
    def safe_div(a, b):
        return a / (b + 1e-8)
    
    # Calculate basic price and volume ratios
    data['close_ratio_1'] = safe_div(data['close'], data['close'].shift(1)) - 1
    data['close_ratio_2'] = safe_div(data['close'].shift(1), data['close'].shift(2)) - 1
    data['close_ratio_3'] = safe_div(data['close'].shift(2), data['close'].shift(3)) - 1
    
    data['high_low_range'] = data['high'] - data['low']
    data['range_ratio_1'] = safe_div(data['high_low_range'], data['high_low_range'].shift(1)) - 1
    data['range_ratio_2'] = safe_div(data['high_low_range'].shift(1), data['high_low_range'].shift(2)) - 1
    
    data['gap_1'] = data['open'] - data['close'].shift(1)
    data['gap_2'] = data['open'].shift(1) - data['close'].shift(2)
    data['close_diff_1'] = data['close'].shift(1) - data['close'].shift(2)
    data['close_diff_2'] = data['close'].shift(2) - data['close'].shift(3)
    
    data['volume_ratio_1'] = safe_div(data['volume'], data['volume'].shift(1)) - 1
    data['volume_ratio_2'] = safe_div(data['volume'].shift(1), data['volume'].shift(2)) - 1
    data['volume_ratio_3'] = safe_div(data['volume'].shift(2), data['volume'].shift(3)) - 1
    
    # Fractal Price-Volume Asymmetry components
    data['fractal_price_momentum_asymmetry'] = (
        data['close_ratio_1'] * data['close_ratio_2'] * data['close_ratio_3'] * 
        np.sign(data['close'] - data['close'].shift(1))
    )
    
    data['price_range_fractal_asymmetry'] = (
        data['range_ratio_1'] * data['range_ratio_2'] * 
        np.sign(data['close'] - data['close'].shift(1))
    )
    
    data['gap_fractal_asymmetry'] = (
        safe_div(data['gap_1'], data['close_diff_1']) * 
        safe_div(data['gap_2'], data['close_diff_2']) * 
        np.sign(data['close'] - data['open'])
    )
    
    # Volume Fractal Asymmetry Dynamics
    data['volume_momentum_fractal_asymmetry'] = (
        data['volume_ratio_1'] * data['volume_ratio_2'] * data['volume_ratio_3'] * 
        np.sign(data['close'] - data['close'].shift(1))
    )
    
    data['volume_spike_fractal_asymmetry'] = (
        safe_div(data['volume'], (data['volume'].shift(1) + data['volume'].shift(2) + data['volume'].shift(3)) / 3) *
        safe_div(data['volume'].shift(1), (data['volume'].shift(2) + data['volume'].shift(3) + data['volume'].shift(4)) / 3) *
        np.sign(data['close'] - data['close'].shift(1))
    )
    
    # Volume Persistence Fractal Asymmetry
    def count_volume_persistence(volume_series, t):
        count = 0
        for i in range(1, 4):
            if volume_series.iloc[t] > volume_series.iloc[t-i]:
                count += 1
        return count
    
    volume_persistence = []
    for i in range(len(data)):
        if i < 4:
            volume_persistence.append(0)
        else:
            count_t = count_volume_persistence(data['volume'], i)
            count_t1 = count_volume_persistence(data['volume'], i-1)
            volume_persistence.append(
                count_t * count_t1 * np.sign(data['close'].iloc[i] - data['close'].iloc[i-1])
            )
    data['volume_persistence_fractal_asymmetry'] = volume_persistence
    
    # Fractal Regime Transition
    data['price_volume_fractal_alignment'] = (
        data['fractal_price_momentum_asymmetry'] * data['volume_momentum_fractal_asymmetry']
    )
    
    data['range_volume_fractal_divergence'] = (
        data['price_range_fractal_asymmetry'] - 
        data['volume_spike_fractal_asymmetry'] * np.sign(data['volume_spike_fractal_asymmetry'])
    )
    
    # Fractal Regime Stability
    def count_alignment(series1, series2, t):
        count = 0
        for i in range(1, 4):
            if t - i >= 0:
                if np.sign(series1.iloc[t-i]) == np.sign(series2.iloc[t-i]):
                    count += 1
        return count
    
    regime_stability = []
    for i in range(len(data)):
        if i < 1:
            regime_stability.append(0)
        else:
            count_align = count_alignment(data['fractal_price_momentum_asymmetry'], 
                                        data['volume_momentum_fractal_asymmetry'], i)
            regime_stability.append(
                count_align * np.sign(data['close'].iloc[i] - data['close'].iloc[i-1])
            )
    data['fractal_regime_stability'] = regime_stability
    
    # Adaptive Fractal Microstructure Dynamics
    data['intraday_flow_fractal_asymmetry'] = (
        safe_div(data['close'] - data['open'], data['high_low_range']) *
        safe_div(data['close'].shift(1) - data['open'].shift(1), data['high_low_range'].shift(1)) *
        np.sign(data['close'] - data['open'])
    )
    
    data['gap_absorption_fractal_asymmetry'] = (
        safe_div(data['gap_1'], data['high_low_range']) *
        safe_div(data['gap_2'], data['high_low_range'].shift(1)) *
        np.sign(data['close'] - data['open'])
    )
    
    data['price_efficiency_fractal_asymmetry'] = (
        (safe_div(data['close'] - data['open'], data['high_low_range']) - 
         safe_div(data['close'].shift(1) - data['open'].shift(1), data['high_low_range'].shift(1))) *
        np.sign(data['close'] - data['close'].shift(1))
    )
    
    # Volume-Price Fractal Integration
    data['volume_adjusted_fractal_momentum'] = (
        data['intraday_flow_fractal_asymmetry'] * 
        safe_div(data['volume'], data['volume'].shift(1)) *
        safe_div(data['volume'].shift(1), data['volume'].shift(2)) *
        np.sign(data['close'] - data['close'].shift(1))
    )
    
    data['volume_range_fractal_collapse'] = (
        (safe_div(data['high'] - data['close'], data['high_low_range']) - 
         safe_div(data['close'] - data['low'], data['high_low_range'])) *
        data['volume_spike_fractal_asymmetry'] *
        np.sign(data['close'] - data['close'].shift(1))
    )
    
    data['fractal_price_volume_divergence'] = (
        data['fractal_price_momentum_asymmetry'] - 
        data['volume_momentum_fractal_asymmetry'] * np.sign(data['volume_momentum_fractal_asymmetry'])
    )
    
    # Adaptive Fractal Synthesis
    data['high_fractal_momentum'] = (
        data['volume_adjusted_fractal_momentum'] * data['volume_spike_fractal_asymmetry']
    )
    
    data['low_fractal_persistence'] = (
        safe_div(data['volume_adjusted_fractal_momentum'], data['volume_spike_fractal_asymmetry'])
    )
    
    data['transition_fractal_momentum'] = (
        data['gap_absorption_fractal_asymmetry'] * data['price_volume_fractal_alignment']
    )
    
    # Multi-Timeframe Fractal Dynamics
    data['microstructure_fractal_absorption'] = (
        safe_div(data['close'] - data['open'], data['gap_1']) *
        data['volume_spike_fractal_asymmetry'] *
        np.sign(data['close'] - data['open'])
    )
    
    data['volume_flow_fractal_collapse'] = (
        (safe_div(data['high'] - data['close'], data['high_low_range']) - 
         safe_div(data['close'] - data['low'], data['high_low_range'])) *
        data['volume_ratio_1'] * data['volume_ratio_2'] *
        np.sign(data['close'] - data['close'].shift(1))
    )
    
    data['range_fractal_momentum'] = (
        data['range_ratio_1'] * data['range_ratio_2'] *
        np.sign(data['close'] - data['close'].shift(1))
    )
    
    # Medium-Term Fractal Integration
    data['fractal_flow_alignment'] = (
        data['microstructure_fractal_absorption'] * data['price_volume_fractal_alignment']
    )
    
    data['fractal_flow_divergence'] = (
        data['volume_flow_fractal_collapse'] * data['volume_momentum_fractal_asymmetry']
    )
    
    data['fractal_range_flow'] = (
        data['range_fractal_momentum'] * data['price_range_fractal_asymmetry']
    )
    
    # Multi-Scale Fractal Synthesis
    data['fractal_flow_momentum'] = (
        data['microstructure_fractal_absorption'] * data['price_efficiency_fractal_asymmetry']
    )
    
    data['fractal_collapse_regime'] = (
        data['volume_flow_fractal_collapse'] * data['fractal_price_volume_divergence']
    )
    
    data['range_fractal_dynamics'] = (
        data['range_fractal_momentum'] * data['volume_momentum_fractal_asymmetry']
    )
    
    # Fractal Signal Coherence
    data['price_volume_fractal_coherence'] = (
        data['price_volume_fractal_alignment'] * data['price_efficiency_fractal_asymmetry']
    )
    
    data['flow_fractal_transition'] = (
        data['fractal_flow_alignment'] * data['transition_fractal_momentum']
    )
    
    data['fractal_flow_stability'] = (
        data['fractal_flow_momentum'] * data['fractal_regime_stability']
    )
    
    # Multi-Scale Fractal Convergence
    data['short_term_fractal_microstructure'] = (
        data['intraday_flow_fractal_asymmetry'] * data['microstructure_fractal_absorption']
    )
    
    data['medium_term_fractal_regime'] = (
        data['price_efficiency_fractal_asymmetry'] * data['range_fractal_momentum']
    )
    
    data['fractal_signal_convergence'] = (
        data['short_term_fractal_microstructure'] * data['medium_term_fractal_regime']
    )
    
    # Adaptive Fractal Weighting
    data['volume_fractal_weighted'] = (
        data['price_volume_fractal_coherence'] * data['volume_adjusted_fractal_momentum']
    )
    
    data['flow_fractal_weighted'] = (
        data['flow_fractal_transition'] * data['fractal_flow_momentum']
    )
    
    data['stability_fractal_weighted'] = (
        data['fractal_flow_stability'] * data['fractal_regime_stability']
    )
    
    # Multi-Scale Fractal Alpha Construction
    data['core_fractal_momentum'] = (
        data['high_fractal_momentum'] * data['low_fractal_persistence']
    )
    
    data['flow_fractal_adaptation'] = (
        data['fractal_signal_convergence'] * data['flow_fractal_weighted']
    )
    
    data['fractal_stability_component'] = (
        data['fractal_flow_stability'] * data['volume_fractal_weighted']
    )
    
    # Final Fractal Alpha
    data['final_fractal_alpha'] = (
        data['core_fractal_momentum'] * 
        data['flow_fractal_adaptation'] * 
        data['fractal_stability_component']
    )
    
    # Return the final alpha factor
    return data['final_fractal_alpha']
