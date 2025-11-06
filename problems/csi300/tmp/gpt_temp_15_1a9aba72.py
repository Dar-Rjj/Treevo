import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Volatility-Volume Reversal Synthesis Alpha Factor
    """
    data = df.copy()
    
    # Volatility-Regime Microstructure Anchoring
    data['vol_gap_asymmetry'] = ((data['open'] - data['close'].shift(1)) / data['close'].shift(1)) * ((data['high'] - data['low']) / data['close'])
    data['vol_intraday_efficiency'] = ((data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)) * (data['volume'] / data['volume'].shift(1))
    data['vol_anchor_divergence'] = ((data['open'] - data['close'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1) + 1e-8)) - ((data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8))
    
    # Volume-Volatility Dislocation Dynamics
    data['vol_range_efficiency'] = ((data['high'] - data['low']) / data['close'].shift(1)) * (data['volume'] / data['volume'].shift(5))
    data['vol_volume_alignment'] = np.sign(data['volume'] / data['volume'].shift(1) - 1) * np.sign((data['high'] - data['low']) / data['close'] - (data['high'].shift(1) - data['low'].shift(1)) / data['close'].shift(1))
    data['vol_volume_dislocation'] = ((data['close'] - data['close'].shift(1)) / (data['high'] - data['low'] + 1e-8)) * (data['volume'] / data['volume'].shift(1))
    
    # Asymmetric Volatility Reversal Patterns
    vol_condition = (data['high'] - data['low']) / data['close'] > (data['high'].shift(1) - data['low'].shift(1)) / data['close'].shift(1)
    data['high_vol_reversal'] = np.where(vol_condition, 
                                        (data['volume'] / data['volume'].shift(1)) * ((data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)), 
                                        0)
    
    low_vol_condition = (data['high'] - data['low']) / data['close'] < (data['high'].shift(1) - data['low'].shift(1)) / data['close'].shift(1)
    data['low_vol_momentum'] = np.where(low_vol_condition, 
                                       (data['volume'] / data['amount']) * (np.abs(data['close'] - data['open']) / data['volume']), 
                                       0)
    
    data['reversal_consistency'] = (np.sign((data['close'] - data['close'].shift(1)) / (data['high'] - data['low'] + 1e-8)) * 
                                   np.sign((data['open'] - data['close'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1) + 1e-8)) * 
                                   np.sign((data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)))
    
    # Cross-Scale Volatility Momentum
    data['vol_weighted_short'] = ((data['close'] / data['close'].shift(1) - 1) / ((data['high'] - data['low']) / data['close'] + 1e-8)) * (data['volume'] / data['volume'].shift(1))
    data['vol_confirmed_medium'] = ((data['close'] / data['close'].shift(3) - 1) * (data['volume'] / data['volume'].shift(1) - 1) * (data['high'] - data['low']) / data['close'])
    data['vol_persistence_div'] = ((data['high'] - data['low']) / data['close'] / ((data['high'].shift(1) - data['low'].shift(1)) / data['close'].shift(1) + 1e-8)) - 1
    
    # Volatility Asymmetry Regime Detection
    data['up_move_efficiency'] = (data['high'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    data['down_move_efficiency'] = (data['open'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    data['vol_asymmetry_ratio'] = data['up_move_efficiency'] / (data['down_move_efficiency'] + 0.001)
    
    high_vol_asym_cond = (data['high'] - data['open']) > (data['open'] - data['low'])
    data['high_vol_volume_asym'] = np.where(high_vol_asym_cond, data['volume'], 0)
    data['low_vol_volume_asym'] = np.where(~high_vol_asym_cond, data['volume'], 0)
    data['vol_volume_asym_ratio'] = data['high_vol_volume_asym'] / (data['low_vol_volume_asym'] + 0.001)
    
    # Volatility Signal Convergence
    data['vol_reversal_alignment'] = ((data['close'] - data['close'].shift(1)) / (data['high'] - data['low'] + 1e-8)) * (data['volume'] / data['volume'].shift(1))
    data['anchor_vol_convergence'] = data['vol_anchor_divergence'] * data['vol_volume_asym_ratio']
    data['flow_vol_integration'] = ((data['amount'] / data['volume']) / (data['high'] - data['low'] + 1e-8)) * (data['volume'] / data['volume'].shift(1)) * data['reversal_consistency']
    
    data['asym_vol_convergence'] = (data['vol_asymmetry_ratio'] * 
                                   ((data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)) * 
                                   ((data['open'] - data['close'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1) + 1e-8)))
    data['vol_flow_alignment'] = data['vol_volume_asym_ratio'] * ((data['amount'] / data['volume']) / (data['high'] - data['low'] + 1e-8))
    data['range_vol_anchor'] = ((data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)) * data['vol_anchor_divergence']
    
    # Dynamic Volatility Factor Construction
    data['vol_divergence_reversal'] = (((data['high'] - data['low']) / data['close'] / ((data['high'].shift(1) - data['low'].shift(1)) / data['close'].shift(1) + 1e-8) - 1) - 
                                      (((data['close'] - data['close'].shift(1)) / (data['high'] - data['low'] + 1e-8)) - 
                                       ((data['open'] - data['close'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1) + 1e-8))))
    
    data['vol_asymmetry_reversal'] = data['vol_asymmetry_ratio'] * data['vol_volume_asym_ratio']
    data['vol_efficiency_reversal'] = (((data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)) * 
                                      ((data['open'] - data['close'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1) + 1e-8)) * 
                                      ((data['amount'] / data['volume']) / (data['high'] - data['low'] + 1e-8)))
    data['vol_anchor_reversal'] = data['vol_anchor_divergence'] * data['reversal_consistency']
    
    # Volatility Regime-Adaptive Weighting
    data['vol_asymmetry_weight'] = 1 + 0.3 * np.sign(data['vol_asymmetry_ratio'] - 1)
    data['vol_volume_weight'] = 1 + 0.2 * np.sign(data['volume'] / data['volume'].shift(1) - 1)
    data['vol_flow_weight'] = 1 + 0.1 * np.sign(((data['amount'] / data['volume']) / (data['high'] - data['low'] + 1e-8)) - 
                                               ((data['amount'].shift(1) / data['volume'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1) + 1e-8)))
    
    # Volatility Factor Confirmation
    data['vol_primary_conf'] = data['vol_divergence_reversal'] * data['vol_asymmetry_reversal']
    data['vol_secondary_conf'] = data['vol_efficiency_reversal'] * data['vol_anchor_reversal']
    data['vol_overall_conf'] = data['vol_primary_conf'] * data['vol_secondary_conf']
    
    # Multi-Scale Volatility Momentum Integration
    data['short_term_vol_momentum'] = (((data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1) + 1e-8)) * 
                                      (data['volume'] / data['volume'].shift(1)) - 
                                      ((data['high'].shift(1) - data['low'].shift(1)) / (data['high'].shift(2) - data['low'].shift(2) + 1e-8)) * 
                                      (data['volume'].shift(1) / data['volume'].shift(2)))
    
    data['medium_term_vol_momentum'] = (((data['high'] - data['low']) / (data['high'].shift(3) - data['low'].shift(3) + 1e-8)) * 
                                       (data['volume'] / data['volume'].shift(3)) - 
                                       ((data['high'].shift(3) - data['low'].shift(3)) / (data['high'].shift(6) - data['low'].shift(6) + 1e-8)) * 
                                       (data['volume'].shift(3) / data['volume'].shift(6)))
    
    # Volatility Momentum Pattern Analysis
    data['vol_acceleration'] = ((data['short_term_vol_momentum'] - data['medium_term_vol_momentum']) * 
                               np.sign(data['volume'] / data['volume'].shift(1) - 1))
    
    # Calculate volatility persistence (rolling count)
    vol_persistence = []
    for i in range(len(data)):
        if i < 3:
            vol_persistence.append(0)
        else:
            count_up = sum(data['short_term_vol_momentum'].iloc[j] > data['short_term_vol_momentum'].iloc[j-1] 
                          for j in range(max(0, i-3), i+1) if j > 0)
            count_down = sum(data['short_term_vol_momentum'].iloc[j] < data['short_term_vol_momentum'].iloc[j-1] 
                           for j in range(max(0, i-3), i+1) if j > 0)
            vol_persistence.append(count_up - count_down)
    
    data['vol_persistence'] = vol_persistence
    
    # Volatility divergence with safe division
    data['vol_divergence'] = np.where((data['short_term_vol_momentum'] != 0) & (data['medium_term_vol_momentum'] != 0),
                                     data['short_term_vol_momentum'] / data['medium_term_vol_momentum'], 0)
    
    # Composite Alpha Synthesis
    data['high_vol_asym_alpha'] = data['vol_divergence_reversal'] * data['vol_efficiency_reversal'] * data['vol_asymmetry_weight']
    data['high_vol_volume_alpha'] = data['vol_asymmetry_reversal'] * data['vol_anchor_reversal'] * data['vol_volume_weight']
    data['high_vol_flow_alpha'] = data['vol_efficiency_reversal'] * data['vol_anchor_reversal'] * data['vol_flow_weight']
    
    # Volatility Momentum Enhancement
    momentum_condition = data['short_term_vol_momentum'] > (data['short_term_vol_momentum'].shift(1) + data['short_term_vol_momentum'].shift(2)) / 2
    data['vol_momentum_core'] = np.where(momentum_condition,
                                        data['high_vol_volume_alpha'] * data['vol_persistence'],
                                        data['high_vol_flow_alpha'] * (1 - data['vol_persistence'] / 2))
    
    data['vol_momentum_enhanced'] = data['vol_momentum_core'] * (1 + np.abs(data['vol_divergence']))
    data['vol_efficiency_weighted'] = data['vol_momentum_enhanced'] * ((data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8))
    
    # Volatility Risk Adjustment
    data['vol_range_adjusted'] = data['vol_efficiency_weighted'] / (data['high'] - data['low'] + 1e-8)
    data['vol_volume_confirmed'] = data['vol_range_adjusted'] * (data['volume'] / data['volume'].shift(1))
    data['vol_efficiency_filter'] = (data['vol_volume_confirmed'] * 
                                    ((data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)) * 
                                    ((data['open'] - data['close'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1) + 1e-8)))
    
    # Final Alpha Output
    data['raw_vol_alpha'] = data['high_vol_asym_alpha'] * data['high_vol_volume_alpha'] * data['high_vol_flow_alpha']
    
    # Confirmed alpha with sign conditions
    sign_conditions = (np.sign(data['vol_persistence']) * 
                      np.sign(data['short_term_vol_momentum']) * 
                      np.sign(data['medium_term_vol_momentum']))
    
    data['confirmed_vol_alpha'] = data['raw_vol_alpha'] * data['vol_overall_conf'] * sign_conditions
    
    # Return the final alpha factor
    return data['confirmed_vol_alpha']
