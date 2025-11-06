import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Multi-Scale Regime Detection
    data['volatility_regime'] = ((data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1))) * np.sign(data['close'] - data['close'].shift(1))
    data['volume_regime'] = (data['volume'] / data['volume'].shift(1)) * np.sign(data['close'] - data['close'].shift(1))
    data['price_momentum_regime'] = (data['close'] - data['close'].shift(1)) / abs(data['close'].shift(1) - data['close'].shift(2)).replace(0, np.nan)
    data['gap_efficiency_regime'] = (abs(data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)) * (data['volume'] / data['volume'].shift(1))
    
    # Volatility Gap Microstructure Patterns
    data['volatility_gap_fracture'] = ((data['close'] - data['close'].shift(1)) / (data['high'] - data['low']).replace(0, np.nan)) * (data['volume'] / data['volume'].shift(1)) * (abs(data['open'] - data['close'].shift(1)) / data['close'].shift(1).replace(0, np.nan))
    data['volume_gap_asymmetry'] = ((data['volume'] - data['volume'].shift(1)) / (data['volume'] + data['volume'].shift(1)).replace(0, np.nan)) * ((data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan))
    data['gap_position_asymmetry'] = ((data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)) * ((data['high'] - data['close']) / (data['high'] - data['low']).replace(0, np.nan))
    data['gap_range_efficiency'] = (abs(data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)) * (data['volume'] / data['volume'].shift(1))
    
    # Fractal Gap Dynamics Integration
    data['fractal_gap_efficiency'] = (np.log(data['high'] - data['low']) / np.log(abs(data['close'] - data['close'].shift(1)).replace(0, np.nan))) * (data['volume'] / abs(data['close'] - data['close'].shift(1)).replace(0, np.nan))
    data['gap_phase_space_momentum'] = ((data['close'] - data['close'].shift(1)) * (data['close'].shift(1) - data['close'].shift(2))) / ((data['high'] - data['low']).replace(0, np.nan) ** 2)
    data['gap_critical_transition'] = ((data['volume'] / data['volume'].shift(1) - 1)) * ((data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1)).replace(0, np.nan))
    data['gap_chaotic_attractor'] = (data['close'] - (data['high'] + data['low']) / 2) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Multi-Timeframe Gap Pressure Dynamics
    data['short_term_gap_pressure'] = ((data['close'] - data['close'].shift(2)) / (abs(data['close'] - data['close'].shift(1)) + abs(data['close'].shift(1) - data['close'].shift(2))).replace(0, np.nan)) * (data['volume'] / data['volume'].shift(2))
    
    medium_term_vol = data['close'].rolling(window=10).apply(lambda x: sum(abs(x.diff().dropna())), raw=False)
    data['medium_term_gap_pressure'] = ((data['close'] - data['close'].shift(10)) / medium_term_vol.replace(0, np.nan)) * ((data['high'] - data['low']) / abs(data['open'] - data['close'].shift(1)).replace(0, np.nan))
    
    long_term_vol = data['close'].rolling(window=20).apply(lambda x: sum(abs(x.diff().dropna())), raw=False)
    data['long_term_gap_pressure'] = ((data['close'] - data['close'].shift(20)) / long_term_vol.replace(0, np.nan)) * (data['volume'] / data['volume'].shift(20))
    
    # Volatility Gap Regime Classification
    data['vol_ratio_3'] = (data['high'] - data['low']) / (data['high'].shift(3) - data['low'].shift(3)).replace(0, np.nan)
    data['vol_ratio_15'] = (data['high'] - data['low']) / (data['high'].shift(15) - data['low'].shift(15)).replace(0, np.nan)
    
    data['volatility_gap_regime'] = 1.0  # Normal
    data.loc[(data['vol_ratio_3'] > 1.4) & (data['vol_ratio_15'] > 1.2), 'volatility_gap_regime'] = 1.25  # High
    data.loc[(data['vol_ratio_3'] < 0.7) & (data['vol_ratio_15'] < 0.8), 'volatility_gap_regime'] = 0.75  # Low
    
    # Volume Gap Regime Detection
    data['volume_ratio_2'] = data['volume'] / data['volume'].shift(2)
    data['volume_avg_5'] = data['volume'].rolling(window=5).mean()
    data['volume_ratio_avg'] = data['volume'] / data['volume_avg_5']
    
    data['volume_gap_regime'] = 1.0  # Normal
    data.loc[(data['volume_ratio_2'] - 1 > 1.6) & (data['volume_ratio_avg'] > 1.3), 'volume_gap_regime'] = 1.2  # High
    data.loc[(data['volume_ratio_2'] - 1 < 0.6) & (data['volume_ratio_avg'] < 0.7), 'volume_gap_regime'] = 0.8  # Low
    
    # Efficiency Gap Regime Detection
    data['gap_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['gap_asymmetry_ratio'] = (data['high'] - data['close']) / (data['close'] - data['low']).replace(0, np.nan)
    
    data['efficiency_gap_regime'] = 1.0  # Normal
    data.loc[(data['gap_efficiency'] > 0.8) & (data['gap_asymmetry_ratio'] > 1.2), 'efficiency_gap_regime'] = 1.15  # High
    data.loc[(data['gap_efficiency'] < 0.2) & (data['gap_asymmetry_ratio'] < 0.8), 'efficiency_gap_regime'] = 0.85  # Low
    
    # Gap Regime Transition Framework
    data['volume_gap_shift'] = (data['volume'] / data['volume'].shift(1)) - (data['volume'].shift(1) / data['volume'].shift(2))
    data['price_gap_momentum'] = ((data['close'] - data['close'].shift(1)) / (data['high'] - data['low']).replace(0, np.nan)) - ((data['close'].shift(1) - data['close'].shift(2)) / (data['high'].shift(1) - data['low'].shift(1)).replace(0, np.nan))
    
    current_efficiency = (abs(data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)) * (data['volume'] / data['volume'].shift(1))
    prev_efficiency = (abs(data['close'].shift(1) - data['open'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1)).replace(0, np.nan)) * (data['volume'].shift(1) / data['volume'].shift(2))
    data['efficiency_gap_change'] = current_efficiency - prev_efficiency
    
    data['gap_asymmetric_balance'] = (data['gap_position_asymmetry'] - data['volume_gap_asymmetry']) / (data['gap_position_asymmetry'] + data['volume_gap_asymmetry']).replace(0, np.nan)
    
    # Adaptive Gap Alpha Components
    data['regime_gap_core'] = data['volatility_regime'] * data['volume_regime'] * data['price_momentum_regime']
    data['microstructure_gap_core'] = data['volume_gap_asymmetry'] * data['gap_position_asymmetry'] * data['gap_asymmetric_balance']
    data['fractal_gap_core'] = data['fractal_gap_efficiency'] * data['gap_phase_space_momentum'] * data['gap_critical_transition']
    data['pressure_gap_core'] = data['short_term_gap_pressure'] * data['medium_term_gap_pressure'] * data['long_term_gap_pressure']
    data['gap_transition_multiplier'] = data['volume_gap_shift'] * data['price_gap_momentum'] * data['efficiency_gap_change']
    
    # Hierarchical Gap Alpha Construction
    data['base_gap_alpha'] = data['regime_gap_core'] * data['microstructure_gap_core'] * data['fractal_gap_core']
    data['pressure_gap_adjustment'] = data['base_gap_alpha'] * data['pressure_gap_core']
    data['regime_enhanced_gap'] = data['pressure_gap_adjustment'] * ((data['volatility_gap_regime'] + data['volume_gap_regime'] + data['efficiency_gap_regime']) / 3)
    data['transition_refined_gap'] = data['regime_enhanced_gap'] * (1 + data['gap_transition_multiplier'])
    data['final_gap_alpha'] = data['transition_refined_gap'] * data['gap_chaotic_attractor'] * data['volatility_gap_fracture']
    
    # Gap Alpha Risk Refinement
    data['volatility_gap_stability'] = data['final_gap_alpha'] / (data['high'] - data['low']).replace(0, np.nan)
    data['volume_gap_consistency'] = data['final_gap_alpha'] * (data['volume'] / data['volume'].shift(5))
    data['price_gap_efficiency'] = data['final_gap_alpha'] * (abs(data['close'] - data['close'].shift(1)) / (data['high'] - data['low']).replace(0, np.nan))
    
    # Final alpha factor
    alpha = data['final_gap_alpha'].fillna(0)
    
    return alpha
