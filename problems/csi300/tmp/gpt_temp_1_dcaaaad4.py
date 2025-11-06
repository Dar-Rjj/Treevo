import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Volatility-Based Volume Components
    data['vol_adj_vol_eff'] = (data['high'] - data['low']) * data['volume'] / data['amount']
    data['vol_weighted_vol_impact'] = (data['high'] - data['low']) * (data['volume'] - data['volume'].shift(1)) / data['volume'].shift(1)
    data['open_gap_vol_response'] = (data['open'] - data['close'].shift(1)) * data['volume'] / data['amount']
    data['close_vol_vol_absorption'] = (data['close'] - data['open']) * data['volume'] / (data['high'] - data['low'])
    
    # Volume-Price Volatility Patterns
    data['vol_vol_divergence'] = (data['volume'] / data['volume'].shift(1)) - ((data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1)))
    data['price_range_vol_concentration'] = (data['high'] - data['low']) * data['volume'] / data['amount']
    data['intraday_vol_vol_eff'] = (data['close'] - data['open']) * data['volume'] / (data['high'] - data['low'])
    
    # Volume Volatility Persistence
    vol_vol_persistence = pd.Series(0, index=data.index)
    vol_inc = (data['volume'] > data['volume'].shift(1)) & (data['high'] - data['low'] > data['high'].shift(1) - data['low'].shift(1))
    for i in range(1, len(data)):
        if vol_inc.iloc[i]:
            vol_vol_persistence.iloc[i] = vol_vol_persistence.iloc[i-1] + 1
    
    # Volatility-Volume Integration
    data['open_close_vol_asymmetry'] = data['open_gap_vol_response'] - data['close_vol_vol_absorption']
    data['vol_vol_momentum_alignment'] = data['vol_weighted_vol_impact'] * data['vol_vol_divergence']
    data['microstructure_vol_eff'] = data['vol_adj_vol_eff'] * data['price_range_vol_concentration']
    
    # Volume Regime Identification
    data['vol_surge_detection'] = data['volume'] / ((data['volume'].shift(1) + data['volume'].shift(2) + data['volume'].shift(3)) / 3)
    
    vol_regime_duration = pd.Series(0, index=data.index)
    vol_inc_simple = data['volume'] > data['volume'].shift(1)
    for i in range(1, len(data)):
        if vol_inc_simple.iloc[i]:
            vol_regime_duration.iloc[i] = vol_regime_duration.iloc[i-1] + 1
    
    data['vol_regime_strength'] = (data['volume'] / data['volume'].shift(1)) * (data['volume'].shift(1) / data['volume'].shift(2))
    data['vol_regime_transition'] = data['vol_surge_detection'] * data['vol_regime_strength']
    
    # Volatility Regime Dynamics
    data['vol_expansion_detection'] = (data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1))
    
    vol_regime_persistence = pd.Series(0, index=data.index)
    vol_exp = data['high'] - data['low'] > data['high'].shift(1) - data['low'].shift(1)
    for i in range(1, len(data)):
        if vol_exp.iloc[i]:
            vol_regime_persistence.iloc[i] = vol_regime_persistence.iloc[i-1] + 1
    
    data['vol_regime_intensity'] = ((data['high'] - data['low']) / (data['high'].shift(2) - data['low'].shift(2))) * \
                                  ((data['high'].shift(1) - data['low'].shift(1)) / (data['high'].shift(3) - data['low'].shift(3)))
    data['vol_regime_shift'] = data['vol_expansion_detection'] * data['vol_regime_intensity']
    
    # Cross-Regime Switching Framework
    data['vol_vol_regime_breakout'] = data['vol_regime_transition'] * data['vol_regime_shift']
    data['regime_switching_confirmation'] = data['vol_regime_strength'] * data['vol_regime_intensity']
    data['multi_regime_switching_convergence'] = data['vol_vol_regime_breakout'] * data['regime_switching_confirmation']
    
    # Hierarchical Asymmetry Switching Patterns
    data['eff_vol_asymmetry_divergence'] = data['vol_adj_vol_eff'] - data['intraday_vol_vol_eff']
    data['open_vol_asymmetry_alignment'] = data['open_gap_vol_response'] * data['vol_vol_divergence']
    data['close_vol_asymmetry_divergence'] = data['close_vol_vol_absorption'] * data['price_range_vol_concentration']
    
    data['intraday_vol_asymmetry_pattern'] = data['vol_expansion_detection'] * (data['volume'] / data['volume'].shift(1))
    data['multi_day_vol_asymmetry'] = data['vol_expansion_detection'] - (data['high'].shift(1) - data['low'].shift(1)) / (data['high'].shift(2) - data['low'].shift(2))
    
    vol_asymmetry_persistence = pd.Series(0, index=data.index)
    vol_inc_count = pd.Series(0, index=data.index)
    vol_dec_count = pd.Series(0, index=data.index)
    
    for i in range(1, len(data)):
        if data['high'].iloc[i] - data['low'].iloc[i] > data['high'].iloc[i-1] - data['low'].iloc[i-1]:
            vol_inc_count.iloc[i] = vol_inc_count.iloc[i-1] + 1
            vol_dec_count.iloc[i] = 0
        elif data['high'].iloc[i] - data['low'].iloc[i] < data['high'].iloc[i-1] - data['low'].iloc[i-1]:
            vol_dec_count.iloc[i] = vol_dec_count.iloc[i-1] + 1
            vol_inc_count.iloc[i] = 0
        else:
            vol_inc_count.iloc[i] = vol_inc_count.iloc[i-1]
            vol_dec_count.iloc[i] = vol_dec_count.iloc[i-1]
    
    data['vol_asymmetry_persistence'] = vol_inc_count - vol_dec_count
    
    # Volatility-Microstructure Asymmetry Dynamics
    vol_micro_asymmetry_dynamics = data['intraday_vol_asymmetry_pattern'] + data['multi_day_vol_asymmetry'] + data['vol_asymmetry_persistence']
    
    data['vol_vol_asymmetry_switching'] = data['vol_regime_transition'] * vol_micro_asymmetry_dynamics
    data['eff_regime_asymmetry_alignment'] = data['eff_vol_asymmetry_divergence'] * data['multi_regime_switching_convergence']
    data['micro_regime_asymmetry_consistency'] = data['microstructure_vol_eff'] * data['regime_switching_confirmation']
    
    # Adaptive Switching Enhancement
    data['vol_regime_weight'] = 1 + abs(data['vol_regime_strength'])
    data['volatility_regime_weight'] = 1 + abs(data['vol_regime_intensity'])
    data['efficiency_volume_weight'] = 1 + abs(data['vol_adj_vol_eff'])
    data['microstructure_volatility_weight'] = 1 + abs(data['vol_vol_divergence'])
    
    # Core Asymmetry Switching Signals
    primary_vol_asymmetry_switching = data['open_close_vol_asymmetry'] * data['vol_vol_momentum_alignment']
    volatility_micro_asymmetry_switching = vol_micro_asymmetry_dynamics * data['microstructure_vol_eff']
    regime_based_asymmetry_switching = data['vol_vol_asymmetry_switching'] * data['eff_regime_asymmetry_alignment']
    
    # Regime-Weighted Switching Core
    regime_weighted_core = (primary_vol_asymmetry_switching * data['vol_regime_weight'] + 
                           volatility_micro_asymmetry_switching * data['volatility_regime_weight'] + 
                           regime_based_asymmetry_switching * data['efficiency_volume_weight']) / 3
    
    # Dynamic Switching Signal Selection
    alpha_factor = pd.Series(0.0, index=data.index)
    
    for i in range(len(data)):
        if data['vol_regime_strength'].iloc[i] > 1:
            price_vol_asymmetry_boost = data['open_close_vol_asymmetry'].iloc[i] * (1 + abs(data['vol_vol_divergence'].iloc[i]))
            alpha_factor.iloc[i] = price_vol_asymmetry_boost
        else:
            vol_micro_enhancement = vol_micro_asymmetry_dynamics.iloc[i] * (1 + abs(data['vol_regime_strength'].iloc[i]))
            alpha_factor.iloc[i] = vol_micro_enhancement
        
        if data['vol_expansion_detection'].iloc[i] > 1.5:
            regime_asymmetry_magnification = data['vol_vol_asymmetry_switching'].iloc[i] * (1 + data['vol_regime_transition'].iloc[i])
            alpha_factor.iloc[i] *= regime_asymmetry_magnification
        else:
            alpha_factor.iloc[i] *= data['eff_regime_asymmetry_alignment'].iloc[i]
        
        if abs(data['vol_vol_divergence'].iloc[i]) > 0.1:
            alpha_factor.iloc[i] *= data['micro_regime_asymmetry_consistency'].iloc[i]
        else:
            alpha_factor.iloc[i] *= data['open_vol_asymmetry_alignment'].iloc[i]
    
    # Final Alpha Factor Construction
    alpha_factor = alpha_factor * data['multi_regime_switching_convergence']
    alpha_factor = alpha_factor * data['micro_regime_asymmetry_consistency']
    alpha_factor = alpha_factor * data['regime_switching_confirmation']
    
    # Clean infinite values and fill NaN
    alpha_factor = alpha_factor.replace([np.inf, -np.inf], np.nan)
    alpha_factor = alpha_factor.fillna(0)
    
    return alpha_factor
