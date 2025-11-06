import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Initialize all intermediate signals
    data['fracture_momentum_intensity'] = 0.0
    data['gap_fracture_momentum'] = 0.0
    data['fracture_range_momentum'] = 0.0
    data['fracture_volume_spike'] = 0.0
    data['fracture_volume_divergence'] = 0.0
    data['fracture_volume_persistence'] = 0.0
    data['fracture_price_momentum_alignment'] = 0.0
    data['fracture_regime_momentum_shift'] = 0.0
    data['fracture_regime_stability'] = 0.0
    data['elastic_fracture_intraday_momentum'] = 0.0
    data['elastic_fracture_gap_momentum'] = 0.0
    data['elastic_fracture_multi_period_momentum'] = 0.0
    data['elastic_volume_fracture_adjusted'] = 0.0
    data['elastic_volume_fracture_divergence'] = 0.0
    data['elastic_volume_fracture_persistence'] = 0.0
    data['high_fracture_momentum_regime'] = 0.0
    data['low_fracture_momentum_regime'] = 0.0
    data['transition_fracture_momentum'] = 0.0
    data['fracture_price_momentum_coherence'] = 0.0
    data['fracture_transition_signals'] = 0.0
    data['fracture_stability_signals'] = 0.0
    data['short_term_fracture_signal'] = 0.0
    data['medium_term_fracture_signal'] = 0.0
    data['fracture_signal_convergence'] = 0.0
    data['volume_fracture_weighted'] = 0.0
    data['stability_fracture_weighted'] = 0.0
    data['transition_fracture_weighted'] = 0.0
    data['core_fracture_momentum'] = 0.0
    data['fracture_adaptation_component'] = 0.0
    data['fracture_stability_component'] = 0.0
    
    # Calculate rolling statistics
    data['close_std_5'] = data['close'].rolling(window=5, min_periods=5).std()
    data['volume_median_20'] = data['volume'].rolling(window=20, min_periods=20).median()
    
    for i in range(5, len(data)):
        # Price Fracture Momentum Signals
        if i >= 5:
            prev_range = data['high'].iloc[i-1] - data['low'].iloc[i-1]
            if prev_range > 0:
                price_change_abs_sum = sum(abs(data['close'].iloc[i-j] - data['close'].iloc[i-j-1]) for j in range(5))
                if price_change_abs_sum > 0:
                    data.loc[data.index[i], 'fracture_momentum_intensity'] = (
                        abs(data['close'].iloc[i] - data['close'].iloc[i-1]) / prev_range * 
                        (data['close'].iloc[i] - data['close'].iloc[i-5]) / price_change_abs_sum
                    )
                    
                    data.loc[data.index[i], 'gap_fracture_momentum'] = (
                        (data['open'].iloc[i] - data['close'].iloc[i-1]) / data['close'].iloc[i-1] * 
                        (data['close'].iloc[i] - data['close'].iloc[i-5]) / price_change_abs_sum
                    )
            
            if data['close_std_5'].iloc[i] > 0:
                data.loc[data.index[i], 'fracture_range_momentum'] = (
                    (data['high'].iloc[i] - data['low'].iloc[i]) / (prev_range + 1e-8) * 
                    (data['close'].iloc[i] - data['close'].iloc[i-1]) / data['close_std_5'].iloc[i]
                )
        
        # Volume Fracture Anomaly Signals
        if i >= 2:
            avg_prev_volume = (data['volume'].iloc[i-2] + data['volume'].iloc[i-1]) / 2
            if avg_prev_volume > 0 and data['volume_median_20'].iloc[i] > 0:
                data.loc[data.index[i], 'fracture_volume_spike'] = (
                    data['volume'].iloc[i] / (avg_prev_volume + 1e-8) * 
                    data['volume'].iloc[i] / data['volume_median_20'].iloc[i]
                )
        
        if i >= 2:
            vol_change_t = data['volume'].iloc[i] / data['volume'].iloc[i-1] - 1
            vol_change_t1 = data['volume'].iloc[i-1] / data['volume'].iloc[i-2] - 1
            price_diff = data['close'].iloc[i] - data['close'].iloc[i-1]
            data.loc[data.index[i], 'fracture_volume_divergence'] = (
                (vol_change_t - vol_change_t1) * np.sign(price_diff) * 
                (data['volume'].iloc[i] - data['volume'].iloc[i-1]) / (abs(price_diff) + 0.001)
            )
        
        if i >= 5:
            volume_change_abs_sum = sum(abs(data['volume'].iloc[i-j] - data['volume'].iloc[i-j-1]) for j in range(5))
            if volume_change_abs_sum > 0:
                data.loc[data.index[i], 'fracture_volume_persistence'] = (
                    (data['volume'].iloc[i] / data['volume'].iloc[i-1]) * 
                    (data['volume'].iloc[i-1] / data['volume'].iloc[i-2]) * 
                    (data['volume'].iloc[i] - data['volume'].iloc[i-5]) / volume_change_abs_sum
                )
        
        # Fracture-Momentum Transition Dynamics
        data.loc[data.index[i], 'fracture_price_momentum_alignment'] = (
            data['fracture_range_momentum'].iloc[i] * data['fracture_volume_spike'].iloc[i]
        )
        
        data.loc[data.index[i], 'fracture_regime_momentum_shift'] = (
            data['gap_fracture_momentum'].iloc[i] * data['fracture_volume_divergence'].iloc[i]
        )
        
        # Fracture Regime Stability (simplified - count over past 3 days)
        if i >= 7:
            stability_count = 0
            for j in range(3):
                if np.sign(data['fracture_range_momentum'].iloc[i-j]) == np.sign(data['fracture_volume_spike'].iloc[i-j]):
                    stability_count += 1
            data.loc[data.index[i], 'fracture_regime_stability'] = (
                stability_count * data['fracture_momentum_intensity'].iloc[i]
            )
        
        # Elastic Fracture-Momentum Extraction
        current_range = data['high'].iloc[i] - data['low'].iloc[i]
        if current_range > 0:
            data.loc[data.index[i], 'elastic_fracture_intraday_momentum'] = (
                (data['close'].iloc[i] - data['open'].iloc[i]) / (current_range + 1e-8) * 
                data['fracture_range_momentum'].iloc[i] * data['fracture_momentum_intensity'].iloc[i]
            )
            
            data.loc[data.index[i], 'elastic_fracture_gap_momentum'] = (
                (data['open'].iloc[i] - data['close'].iloc[i-1]) / (current_range + 1e-8) * 
                (data['close'].iloc[i] - data['close'].iloc[i-1]) / (current_range + 1e-8) * 
                data['gap_fracture_momentum'].iloc[i]
            )
        
        if i >= 5 and current_range > 0:
            prev_range = data['high'].iloc[i-1] - data['low'].iloc[i-1]
            price_change_abs_sum = sum(abs(data['close'].iloc[i-j] - data['close'].iloc[i-j-1]) for j in range(5))
            if prev_range > 0 and price_change_abs_sum > 0:
                data.loc[data.index[i], 'elastic_fracture_multi_period_momentum'] = (
                    (data['close'].iloc[i] - data['close'].iloc[i-2]) / (current_range + 1e-8) * 
                    (data['close'].iloc[i-1] - data['close'].iloc[i-3]) / (prev_range + 1e-8) * 
                    (data['close'].iloc[i] - data['close'].iloc[i-5]) / price_change_abs_sum
                )
        
        # Volume-Fracture Weighted Momentum
        if i >= 1:
            data.loc[data.index[i], 'elastic_volume_fracture_adjusted'] = (
                data['elastic_fracture_intraday_momentum'].iloc[i] * 
                (data['volume'].iloc[i] / data['volume'].iloc[i-1]) * 
                data['fracture_volume_spike'].iloc[i]
            )
            
            data.loc[data.index[i], 'elastic_volume_fracture_divergence'] = (
                data['elastic_fracture_gap_momentum'].iloc[i] * data['fracture_volume_divergence'].iloc[i]
            )
            
            data.loc[data.index[i], 'elastic_volume_fracture_persistence'] = (
                data['elastic_fracture_multi_period_momentum'].iloc[i] * data['fracture_volume_persistence'].iloc[i]
            )
        
        # Regime-Fracture Momentum Synthesis
        data.loc[data.index[i], 'high_fracture_momentum_regime'] = (
            data['elastic_volume_fracture_adjusted'].iloc[i] * data['fracture_volume_spike'].iloc[i]
        )
        
        data.loc[data.index[i], 'low_fracture_momentum_regime'] = (
            data['elastic_volume_fracture_persistence'].iloc[i] / (data['fracture_volume_spike'].iloc[i] + 1e-8)
        )
        
        data.loc[data.index[i], 'transition_fracture_momentum'] = (
            data['elastic_fracture_gap_momentum'].iloc[i] * data['fracture_regime_momentum_shift'].iloc[i]
        )
        
        # Dynamic Fracture-Momentum Integration
        data.loc[data.index[i], 'fracture_price_momentum_coherence'] = (
            data['fracture_price_momentum_alignment'].iloc[i] * data['elastic_volume_fracture_adjusted'].iloc[i]
        )
        
        data.loc[data.index[i], 'fracture_transition_signals'] = (
            data['fracture_regime_momentum_shift'].iloc[i] * data['transition_fracture_momentum'].iloc[i]
        )
        
        data.loc[data.index[i], 'fracture_stability_signals'] = (
            data['fracture_regime_stability'].iloc[i] * data['elastic_volume_fracture_persistence'].iloc[i]
        )
        
        data.loc[data.index[i], 'short_term_fracture_signal'] = (
            data['elastic_fracture_intraday_momentum'].iloc[i] * data['elastic_volume_fracture_divergence'].iloc[i]
        )
        
        data.loc[data.index[i], 'medium_term_fracture_signal'] = (
            data['elastic_fracture_multi_period_momentum'].iloc[i] * data['elastic_volume_fracture_persistence'].iloc[i]
        )
        
        data.loc[data.index[i], 'fracture_signal_convergence'] = (
            data['short_term_fracture_signal'].iloc[i] * data['medium_term_fracture_signal'].iloc[i]
        )
        
        data.loc[data.index[i], 'volume_fracture_weighted'] = (
            data['fracture_price_momentum_coherence'].iloc[i] * data['elastic_volume_fracture_adjusted'].iloc[i]
        )
        
        data.loc[data.index[i], 'stability_fracture_weighted'] = (
            data['fracture_stability_signals'].iloc[i] * data['fracture_regime_stability'].iloc[i]
        )
        
        data.loc[data.index[i], 'transition_fracture_weighted'] = (
            data['fracture_transition_signals'].iloc[i] * data['transition_fracture_momentum'].iloc[i]
        )
        
        # Dynamic Fracture-Momentum Alpha Construction
        data.loc[data.index[i], 'core_fracture_momentum'] = (
            data['high_fracture_momentum_regime'].iloc[i] * data['low_fracture_momentum_regime'].iloc[i]
        )
        
        data.loc[data.index[i], 'fracture_adaptation_component'] = (
            data['fracture_signal_convergence'].iloc[i] * data['transition_fracture_weighted'].iloc[i]
        )
        
        data.loc[data.index[i], 'fracture_stability_component'] = (
            data['fracture_stability_signals'].iloc[i] * data['volume_fracture_weighted'].iloc[i]
        )
    
    # Final Alpha
    data['final_alpha'] = (
        data['core_fracture_momentum'] * 
        data['fracture_adaptation_component'] * 
        data['fracture_stability_component']
    )
    
    return data['final_alpha']
