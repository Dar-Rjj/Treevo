import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Asymmetric Range-Momentum Framework
    # Range-Based Momentum Acceleration
    data['range_price_accel'] = ((data['close'] - data['close'].shift(1)) / (data['high'] - data['low'])) - \
                               ((data['close'].shift(1) - data['close'].shift(2)) / (data['high'].shift(1) - data['low'].shift(1)))
    
    data['range_width_accel'] = ((data['high'] - data['low']) / (data['high'].shift(3) - data['low'].shift(3))) - 1
    
    data['asymmetric_range_persistence'] = np.sign(data['close'] - data['close'].shift(1)) * \
                                          np.sign(data['close'].shift(1) - data['close'].shift(2)) * \
                                          (data['high'] - data['low'])
    
    # Volatility Asymmetry Integration
    data['upper_range_efficiency'] = (data['high'] - data['open']) / (data['high'] - data['low'])
    data['lower_range_efficiency'] = (data['open'] - data['low']) / (data['high'] - data['low'])
    data['range_volatility_asymmetry'] = data['upper_range_efficiency'] - data['lower_range_efficiency']
    
    # Range-Momentum Alignment
    data['acceleration_range_alignment'] = data['range_price_accel'] * data['range_volatility_asymmetry']
    data['width_range_synchronization'] = data['range_width_accel'] * data['range_volatility_asymmetry']
    data['asymmetric_range_confirmation'] = data['asymmetric_range_persistence'] * data['range_volatility_asymmetry']
    
    # Volume-Range Convergence Framework
    # Volume Range Asymmetry
    data['upper_range_volume_flow'] = data['amount'] * (data['high'] - data['close']) / (data['high'] - data['low'])
    data['lower_range_volume_flow'] = data['amount'] * (data['close'] - data['low']) / (data['high'] - data['low'])
    data['volume_range_asymmetry'] = data['upper_range_volume_flow'] - data['lower_range_volume_flow']
    
    # Volume-Range Price Convergence
    range_directional_volume_imbalance = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 4:
            pos_sum = 0
            neg_sum = 0
            for j in range(5):
                if i - j >= 0 and i - j - 1 >= 0:
                    price_change = (data['close'].iloc[i-j] - data['close'].iloc[i-j-1]) / (data['high'].iloc[i-j] - data['low'].iloc[i-j])
                    if price_change > 0:
                        pos_sum += price_change
                    
                    if i - j - 1 >= 0 and i - j - 2 >= 0:
                        prev_price_change = (data['close'].iloc[i-j-1] - data['close'].iloc[i-j-2]) / (data['high'].iloc[i-j-1] - data['low'].iloc[i-j-1])
                        if prev_price_change > 0:
                            neg_sum += prev_price_change
            
            range_directional_volume_imbalance.iloc[i] = pos_sum - neg_sum
    
    data['range_directional_volume_imbalance'] = range_directional_volume_imbalance
    
    data['volume_range_alignment'] = np.sign(data['volume'] - data['volume'].shift(5)) * \
                                    np.sign((data['close'] - data['close'].shift(3)) / (data['high'] - data['low']))
    
    data['volume_range_convergence_strength'] = data['volume_range_asymmetry'] * data['volume_range_alignment']
    
    # Range Microstructure Efficiency
    data['range_trade_efficiency'] = data['amount'] / data['volume'] * (data['high'] - data['low'])
    data['range_movement_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['range_order_flow'] = (data['close'] - data['open']) * data['volume'] / (data['amount'] * (data['high'] - data['low']))
    
    # Range Fracture Detection
    # Range Price Fracture Components
    data['upper_range_fracture'] = ((data['high'] - data['high'].shift(1)) / (data['high'].shift(1) - data['high'].shift(2))) * (data['high'] - data['low'])
    data['lower_range_fracture'] = ((data['low'] - data['low'].shift(1)) / (data['low'].shift(1) - data['low'].shift(2))) * (data['high'] - data['low'])
    
    data['range_fracture_threshold'] = ((np.abs(data['upper_range_fracture']) > 2.0) | (np.abs(data['lower_range_fracture']) > 2.0)).astype(float)
    
    # Volume-Range Fracture
    data['volume_range_convergence_fracture'] = (data['range_fracture_threshold'] == 1) & \
                                               (np.sign(data['volume_range_asymmetry']) * np.sign(data['range_volatility_asymmetry']) < 0)
    
    data['range_volatility_alignment_fracture'] = (data['range_volatility_asymmetry'] < 0) & (data['range_directional_volume_imbalance'] > 0)
    
    data['multi_range_fracture'] = (data['range_fracture_threshold'] == 1) & \
                                  (np.sign(data['range_volatility_asymmetry']) != np.sign(data['range_volatility_asymmetry'].shift(1)))
    
    # Range Fracture Persistence
    data['range_volatility_persistence'] = np.sign(data['range_volatility_asymmetry']) * \
                                          np.sign(data['range_volatility_asymmetry'].shift(1)) * \
                                          data['range_volatility_asymmetry']
    
    data['range_volume_persistence'] = np.sign(data['volume_range_asymmetry']) * \
                                      np.sign(data['volume_range_asymmetry'].shift(1)) * \
                                      data['volume_range_convergence_strength']
    
    data['enhanced_range_fracture'] = data['volume_range_convergence_fracture'] & (data['range_volatility_persistence'] > 0)
    
    # Range Convergence Regime Analysis
    # Asymmetric Range Regime Detection
    data['upper_range_convergence_regime'] = (data['range_volatility_asymmetry'] > 0) & (data['volume_range_asymmetry'] > 0)
    data['lower_range_convergence_regime'] = (data['range_volatility_asymmetry'] < 0) & (data['volume_range_asymmetry'] < 0)
    data['mixed_range_convergence_regime'] = ~(data['upper_range_convergence_regime'] | data['lower_range_convergence_regime'])
    
    # Multi-Range Signal Alignment
    data['short_range_alignment'] = data['range_movement_efficiency'] * data['range_directional_volume_imbalance'] * data['volume_range_alignment']
    data['medium_range_alignment'] = data['range_width_accel'] * data['asymmetric_range_confirmation'] * data['range_volume_persistence']
    data['range_signal_convergence'] = np.sign(data['short_range_alignment']) * np.sign(data['medium_range_alignment'])
    
    # Range Regime Strength Assessment
    data['range_volatility_regime_strength'] = np.abs(data['range_volatility_asymmetry']) * np.abs(data['range_width_accel'])
    data['range_volume_regime_strength'] = np.abs(data['volume_range_asymmetry']) * np.abs(data['volume_range_convergence_strength'])
    data['range_convergence_strength'] = data['range_volatility_regime_strength'] * data['range_volume_regime_strength']
    
    # Factor Integration
    # Core Range Convergence Signals
    data['range_momentum_volatility_core'] = data['acceleration_range_alignment'] * data['asymmetric_range_confirmation']
    data['volume_range_convergence_core'] = data['volume_range_convergence_strength'] * data['range_order_flow']
    data['multi_range_core'] = data['range_signal_convergence'] * data['range_convergence_strength']
    
    # Range Fracture Enhancement
    data['immediate_range_fracture'] = data['multi_range_fracture'] * data['acceleration_range_alignment']
    data['sustained_range_fracture'] = data['enhanced_range_fracture'] * data['range_signal_convergence']
    data['range_fracture_intensity'] = data['immediate_range_fracture'] * data['sustained_range_fracture']
    
    # Range Regime-Adaptive Construction
    data['upper_range_regime_factor'] = data['upper_range_convergence_regime'] * data['range_momentum_volatility_core'] * data['volume_range_convergence_core']
    data['lower_range_regime_factor'] = data['lower_range_convergence_regime'] * data['range_momentum_volatility_core'] * data['volume_range_convergence_core']
    data['mixed_range_regime_factor'] = data['mixed_range_convergence_regime'] * data['multi_range_core'] * data['range_fracture_intensity']
    
    # Final Alpha Generation
    # Range Regime-Weighted Components
    data['upper_range_convergence_alpha'] = data['upper_range_regime_factor'] * data['range_volatility_regime_strength']
    data['lower_range_convergence_alpha'] = data['lower_range_regime_factor'] * data['range_volume_regime_strength']
    data['mixed_range_convergence_alpha'] = data['mixed_range_regime_factor'] * data['range_convergence_strength']
    
    # Range Fracture Signal Integration
    data['fracture_enhanced_upper_range'] = data['upper_range_convergence_alpha'] * (1 + data['immediate_range_fracture'])
    data['fracture_enhanced_lower_range'] = data['lower_range_convergence_alpha'] * (1 + data['sustained_range_fracture'])
    data['fracture_enhanced_mixed_range'] = data['mixed_range_convergence_alpha'] * (1 + data['range_fracture_intensity'])
    
    # Composite Range Alpha Factor
    data['high_range_convergence_signal'] = data['fracture_enhanced_upper_range'] + data['fracture_enhanced_lower_range']
    data['adaptive_mixed_range_signal'] = data['fracture_enhanced_mixed_range'] * data['range_signal_convergence']
    
    # Final Asymmetric Range-Momentum Volatility Convergence Factor
    factor = data['high_range_convergence_signal'] + data['adaptive_mixed_range_signal']
    
    return factor
