import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize all intermediate columns
    data['prev_close'] = data['close'].shift(1)
    data['prev_high'] = data['high'].shift(1)
    data['prev_low'] = data['low'].shift(1)
    data['prev_volume'] = data['volume'].shift(1)
    data['prev_amount'] = data['amount'].shift(1)
    data['prev_open'] = data['open'].shift(1)
    data['prev2_close'] = data['close'].shift(2)
    data['prev2_high'] = data['high'].shift(2)
    data['prev2_low'] = data['low'].shift(2)
    
    # Price Fracture Stress Detection
    data['opening_fracture_stress'] = ((data['open'] - data['prev_close']) / 
                                      (data['prev_high'] - data['prev_low'] + 1e-8) * 
                                      np.abs(data['close'] - data['open']) / 
                                      (data['high'] - data['low'] + 1e-8))
    
    data['midday_fracture_stress'] = (np.abs((data['high'] + data['low'])/2 - (data['open'] + data['close'])/2) / 
                                     (data['high'] - data['low'] + 1e-8) * 
                                     (data['high'] - data['low']) / 
                                     (np.abs(data['close'] - data['prev_close']) + 1e-8))
    
    data['closing_fracture_stress'] = ((data['close'] - (data['high'] + data['low'])/2) * 
                                      np.sign(data['close'] - data['open']) / 
                                      (data['high'] - data['low'] + 1e-8) * 
                                      data['volume'])
    
    # Volume Fracture Stress Dynamics
    volume_diff = data['volume'] - data['prev_volume']
    data['volume_fracture_intensity'] = (volume_diff * 
                                        (data['close'] - data['prev_close']) / 
                                        (np.abs(volume_diff) + 1e-8) * 
                                        np.abs(data['close'] - data['open']) / 
                                        (data['high'] - data['low'] + 1e-8))
    
    amount_diff = data['amount'] - data['prev_amount']
    data['flow_fracture_stress'] = (amount_diff * 
                                   (data['close'] - data['prev_close']) / 
                                   (np.abs(amount_diff) + 1e-8) * 
                                   (data['high'] - data['low']) / 
                                   (np.abs(data['close'] - data['prev_close']) + 1e-8))
    
    data['volume_flow_fracture_divergence'] = (data['volume_fracture_intensity'] - 
                                              data['flow_fracture_stress'])
    
    # Fracture-Stress Momentum Patterns
    data['fracture_stress_intensity'] = (np.abs(data['open'] - data['prev_close']) / 
                                        (data['prev_high'] - data['prev_low'] + 1e-8) - 
                                        np.abs(data['prev_open'] - data['prev2_close']) / 
                                        (data['prev2_high'] - data['prev2_low'] + 1e-8))
    
    data['stress_momentum_discontinuity'] = (np.abs(data['close'] - data['open']) / 
                                            (data['high'] - data['low'] + 1e-8) - 
                                            np.abs(data['prev_close'] - data['prev_open']) / 
                                            (data['prev_high'] - data['prev_low'] + 1e-8) * 
                                            (data['high'] - data['low']) / 
                                            (np.abs(data['close'] - data['prev_close']) + 1e-8))
    
    # Rejection-Fracture Asymmetry System
    data['upper_fracture_rejection'] = ((data['high'] - np.maximum(data['open'], data['close'])) / 
                                       (data['high'] - data['low'] + 1e-8) * 
                                       data['volume'] * 
                                       data['opening_fracture_stress'])
    
    data['lower_fracture_rejection'] = ((np.minimum(data['open'], data['close']) - data['low']) / 
                                       (data['high'] - data['low'] + 1e-8) * 
                                       data['volume'] * 
                                       data['closing_fracture_stress'])
    
    data['net_fracture_rejection'] = (data['upper_fracture_rejection'] - 
                                     data['lower_fracture_rejection'])
    
    # Multi-Timeframe Fracture Rejection
    data['three_day_high_fracture_rejection'] = ((data['high'] - np.maximum(data['prev2_close'], 
                                                                           data['prev_close'], 
                                                                           data['close'])) / 
                                                (data['high'] - data['low'] + 1e-8) * 
                                                data['volume'] * 
                                                data['midday_fracture_stress'])
    
    data['three_day_low_fracture_rejection'] = ((np.minimum(data['prev2_close'], 
                                                           data['prev_close'], 
                                                           data['close']) - data['low']) / 
                                               (data['high'] - data['low'] + 1e-8) * 
                                               data['volume'] * 
                                               data['closing_fracture_stress'])
    
    data['net_multi_timeframe_fracture_rejection'] = ((data['three_day_high_fracture_rejection'] - 
                                                      data['three_day_low_fracture_rejection']) * 
                                                     np.sign(data['close'] - data['open']))
    
    # Microstructure Stress Efficiency Coupling
    data['volume_fracture_efficiency'] = (data['volume'] * 
                                         np.abs(data['close'] - data['open']) / 
                                         (data['high'] - data['low'] + 1e-8) * 
                                         data['volume_fracture_intensity'])
    
    data['prev_volume_fracture_efficiency'] = data['volume_fracture_efficiency'].shift(1)
    data['volume_stress_fracture_momentum'] = (data['volume_fracture_efficiency'] / 
                                              (data['prev_volume_fracture_efficiency'] + 1e-8) - 1)
    
    data['fracture_efficiency_volume_coherence'] = (data['volume_fracture_efficiency'] * 
                                                   data['volume_fracture_intensity'])
    
    # Spread-Fracture Stress Interaction
    data['effective_fracture_spread'] = (2 * np.abs(data['close'] - (data['high'] + data['low'])/2) / 
                                        ((data['high'] + data['low'])/2 + 1e-8))
    
    data['spread_fracture_stress'] = (data['effective_fracture_spread'] * 
                                     np.abs(data['close'] - data['open']) / 
                                     (data['high'] - data['low'] + 1e-8) * 
                                     (data['high'] - data['low']) / 
                                     (np.abs(data['close'] - data['prev_close']) + 1e-8))
    
    data['prev_spread_fracture_stress'] = data['spread_fracture_stress'].shift(1)
    data['spread_fracture_stress_momentum'] = (data['spread_fracture_stress'] / 
                                              (data['prev_spread_fracture_stress'] + 1e-8) - 1)
    
    # Fracture-Stress Validation and Persistence
    data['price_fracture_stress_divergence'] = (np.sign(data['close']/data['prev_close'] - 1) * 
                                               np.sign(data['fracture_stress_intensity'] - 
                                                      data['fracture_stress_intensity'].shift(1)))
    
    data['volume_fracture_stress_divergence'] = (np.sign(data['volume']/data['prev_volume'] - 1) * 
                                                np.sign(data['volume_fracture_intensity'] - 
                                                       data['volume_fracture_intensity'].shift(1)))
    
    data['rejection_fracture_stress_divergence'] = (np.sign(data['net_fracture_rejection'] - 
                                                           data['net_fracture_rejection'].shift(1)) * 
                                                   np.sign(data['volume_fracture_efficiency'] - 
                                                          data['volume_fracture_efficiency'].shift(1)))
    
    # Calculate persistence metrics using rolling windows
    def calculate_persistence(series, window=3):
        result = pd.Series(index=series.index, dtype=float)
        for i in range(window-1, len(series)):
            window_data = series.iloc[i-window+1:i+1]
            if len(window_data) >= 3:
                signs = []
                for j in range(1, len(window_data)):
                    signs.append(np.sign(window_data.iloc[j] - window_data.iloc[j-1]))
                count = 0
                for k in range(1, len(signs)):
                    if signs[k] == signs[k-1]:
                        count += 1
                result.iloc[i] = count / (len(signs) - 1)
        return result
    
    data['fracture_stress_persistence'] = calculate_persistence(data['fracture_stress_intensity'])
    data['rejection_fracture_persistence'] = calculate_persistence(data['net_fracture_rejection'])
    data['volume_fracture_persistence'] = calculate_persistence(data['volume'] / data['volume'].shift(1) - 1)
    
    # Fracture Stress Confirmation
    data['opening_fracture_stress_confirmation'] = (data['opening_fracture_stress'] * 
                                                   np.sign(data['open'] - data['prev_close']))
    
    data['volume_fracture_stress_confirmation'] = (data['volume_fracture_intensity'] * 
                                                  np.sign(data['volume'] - data['prev_volume']))
    
    data['stress_momentum_confirmation'] = (data['stress_momentum_discontinuity'] * 
                                           np.sign(np.abs(data['close'] - data['open'])/(data['high'] - data['low'] + 1e-8) - 
                                                  np.abs(data['prev_close'] - data['prev_open'])/(data['prev_high'] - data['prev_low'] + 1e-8)))
    
    # Regime-Based Fracture Stress Classification
    data['fracture_stress_regime'] = np.where(data['fracture_stress_intensity'] > 0.02, 'high',
                                             np.where(data['fracture_stress_intensity'] < -0.02, 'low', 'moderate'))
    
    # Adaptive Fracture Stress Components
    data['rejection_fracture_stress_asymmetry'] = (data['net_multi_timeframe_fracture_rejection'] * 
                                                  data['volume_stress_fracture_momentum'])
    
    data['fracture_enhanced_rejection_stress'] = (data['net_fracture_rejection'] * 
                                                 data['fracture_stress_intensity'])
    
    data['volume_spread_fracture_stress'] = (data['fracture_efficiency_volume_coherence'] * 
                                            data['spread_fracture_stress_momentum'])
    
    # Validation-Enhanced Fracture Signals
    data['validated_rejection_fracture_asymmetry'] = (data['rejection_fracture_stress_asymmetry'] * 
                                                     data['rejection_fracture_stress_divergence'])
    
    data['fracture_validated_stress_efficiency'] = (data['fracture_enhanced_rejection_stress'] * 
                                                   data['opening_fracture_stress_confirmation'])
    
    data['volume_validated_spread_fracture'] = (data['volume_spread_fracture_stress'] * 
                                               data['volume_fracture_stress_divergence'])
    
    # Regime-Adaptive Fracture Weighting
    regime_weights = np.where(data['fracture_stress_regime'] == 'high', 1.3,
                             np.where(data['fracture_stress_regime'] == 'low', 1.2, 1.0))
    
    data['fracture_persistence_weight'] = data['fracture_stress_persistence']
    
    # Apply regime weights to components
    data['validated_rejection_fracture_asymmetry'] *= regime_weights
    data['fracture_validated_stress_efficiency'] *= regime_weights
    data['volume_validated_spread_fracture'] *= regime_weights
    
    # Final Fracture-Stress Alpha Synthesis
    data['primary_factor'] = (data['validated_rejection_fracture_asymmetry'] * 
                             data['fracture_persistence_weight'])
    
    data['secondary_factor'] = (data['fracture_validated_stress_efficiency'] * 
                               data['rejection_fracture_persistence'])
    
    data['tertiary_factor'] = (data['volume_validated_spread_fracture'] * 
                              data['volume_fracture_persistence'])
    
    # Composite Alpha
    alpha = (data['primary_factor'] * 
             data['secondary_factor'] * 
             data['tertiary_factor'] * 
             data['stress_momentum_confirmation'])
    
    # Clean up intermediate columns and return only the alpha series
    alpha_series = alpha.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return alpha_series
