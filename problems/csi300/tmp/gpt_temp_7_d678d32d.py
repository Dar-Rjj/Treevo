import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Fractal Flow Regime Classification
    # Volatility-Flow Components
    data['daily_range_vol'] = (data['high'] - data['low']) / data['close']
    data['flow_vol_momentum'] = data['daily_range_vol'] - data['daily_range_vol'].shift(1)
    data['vol_flow_divergence'] = (data['volume'] / (data['high'] - data['low'] + 1e-8)) - \
                                 (data['volume'].shift(1) / (data['high'].shift(1) - data['low'].shift(1) + 1e-8))
    
    # Multi-Scale Flow Asymmetry
    data['micro_flow_asym'] = ((data['high'] - data['open']) / (data['high'] - data['low'] + 1e-8)) - \
                             ((data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8))
    
    # Rolling calculations for meso and macro flow asymmetry
    data['high_3d'] = data['high'].rolling(window=3, min_periods=3).max()
    data['low_3d'] = data['low'].rolling(window=3, min_periods=3).min()
    data['meso_flow_asym'] = ((data['high_3d'] - data['open']) / (data['high_3d'] - data['low_3d'] + 1e-8)) - \
                            ((data['open'] - data['low_3d']) / (data['high_3d'] - data['low_3d'] + 1e-8))
    
    data['high_8d'] = data['high'].rolling(window=8, min_periods=8).max()
    data['low_8d'] = data['low'].rolling(window=8, min_periods=8).min()
    data['macro_flow_asym'] = ((data['high_8d'] - data['open']) / (data['high_8d'] - data['low_8d'] + 1e-8)) - \
                             ((data['open'] - data['low_8d']) / (data['high_8d'] - data['low_8d'] + 1e-8))
    
    # Adaptive Regime Framework
    data['high_flow_regime'] = (data['daily_range_vol'] > data['daily_range_vol'].shift(1)) & \
                              (data['vol_flow_divergence'] > 0)
    data['low_flow_regime'] = (data['daily_range_vol'] < data['daily_range_vol'].shift(1)) & \
                             (data['vol_flow_divergence'] < 0)
    data['fractal_flow_expansion'] = (data['micro_flow_asym'] > data['meso_flow_asym']) & \
                                    (data['meso_flow_asym'] > data['macro_flow_asym'])
    data['fractal_flow_contraction'] = (data['micro_flow_asym'] < data['meso_flow_asym']) & \
                                      (data['meso_flow_asym'] < data['macro_flow_asym'])
    data['transition_flow_regime'] = ~data['high_flow_regime'] & ~data['low_flow_regime']
    
    # Multi-Scale Range Efficiency
    data['micro_range_eff'] = (data['close'] - data['close'].shift(1)) / (data['high'] - data['low'] + 1e-8)
    
    data['high_5d'] = data['high'].rolling(window=5, min_periods=5).max()
    data['low_5d'] = data['low'].rolling(window=5, min_periods=5).min()
    data['meso_range_eff'] = (data['close'] - data['close'].shift(5)) / (data['high_5d'] - data['low_5d'] + 1e-8)
    
    data['high_13d'] = data['high'].rolling(window=13, min_periods=13).max()
    data['low_13d'] = data['low'].rolling(window=13, min_periods=13).min()
    data['macro_range_eff'] = (data['close'] - data['close'].shift(13)) / (data['high_13d'] - data['low_13d'] + 1e-8)
    
    data['fractal_eff_cascade'] = data['micro_range_eff'] * data['meso_range_eff'] * data['macro_range_eff']
    
    # Volume-Pressure Dynamics
    data['buy_pressure_vol'] = data['volume'] * (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    data['sell_pressure_vol'] = data['volume'] * (data['high'] - data['close']) / (data['high'] - data['low'] + 1e-8)
    data['pressure_ratio'] = data['buy_pressure_vol'] / (data['sell_pressure_vol'] + 1e-8)
    data['pressure_asymmetry'] = (data['pressure_ratio'] - data['pressure_ratio'].shift(1)) * \
                                np.sign(data['buy_pressure_vol'] - data['sell_pressure_vol'])
    
    # Flow Efficiency Integration
    data['vol_weighted_eff'] = data['fractal_eff_cascade'] * data['pressure_asymmetry']
    data['range_momentum'] = (data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1) + 1e-8) - 1
    data['efficiency_momentum'] = data['fractal_eff_cascade'] - \
                                 (data['micro_range_eff'] * data['meso_range_eff'] * \
                                  np.sign(data['micro_range_eff'] - data['meso_range_eff']))
    
    # Flow Transmission Framework
    # Multi-Timeframe Flow Pressure
    data['intraday_flow_pressure'] = ((data['open'] - data['close'].shift(1)) / (data['close'].shift(1) + 1e-8)) * \
                                    ((data['close'] - data['open']) / (data['open'] + 1e-8))
    
    data['range_flow_pressure'] = ((data['close'] - data['low_3d']) / (data['high_3d'] - data['low_3d'] + 1e-8)) * \
                                 ((data['close'] - data['close'].shift(3)) / (data['close'].shift(3) + 1e-8))
    
    data['flow_pressure_convergence'] = data['intraday_flow_pressure'] * data['range_flow_pressure']
    
    # Volume Flow Dynamics
    data['vol_flow_micro'] = (data['volume'] / (data['volume'].shift(1) + 1e-8)) * \
                            (np.abs(data['volume'] - data['volume'].shift(1)) / (data['volume'] + 1e-8))
    
    data['vol_flow_meso'] = (data['volume'] / (data['volume'].shift(5) + 1e-8)) * \
                           (np.abs(data['volume'] - data['volume'].shift(5)) / (data['volume'] + 1e-8))
    
    data['vol_flow_macro'] = (data['volume'] / (data['volume'].shift(13) + 1e-8)) * \
                            (np.abs(data['volume'] - data['volume'].shift(13)) / (data['volume'] + 1e-8))
    
    data['vol_flow_cascade'] = data['vol_flow_micro'] * data['vol_flow_meso'] * data['vol_flow_macro'] * \
                              np.sign(data['vol_flow_micro'] - data['vol_flow_meso'])
    
    # Flow Transmission Integration
    data['flow_direction_alignment'] = np.sign(data['close'] - data['close'].shift(1)) * \
                                      np.sign(data['volume'] - data['volume'].shift(1))
    
    data['value_flow_ratio'] = ((data['amount'] / (data['amount'].shift(1) + 1e-8)) / \
                               (data['volume'] / (data['volume'].shift(1) + 1e-8)))
    
    data['flow_transmission_core'] = data['vol_weighted_eff'] * data['flow_pressure_convergence'] * data['vol_flow_cascade']
    
    # Regime-Adaptive Signal Construction
    # High Flow Efficiency Signals
    data['flow_breakout_eff'] = ((data['close'] - data['close'].shift(1)) / (data['high'] - data['low'] + 1e-8)) * \
                               (data['volume'] / (data['volume'].shift(1) + 1e-8))
    
    data['flow_gap_absorption'] = ((data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)) * \
                                 np.sign(data['open'] - data['close'].shift(1))
    
    data['high_flow_core'] = data['flow_breakout_eff'] * data['flow_gap_absorption'] * data['micro_range_eff']
    
    # Low Flow Efficiency Signals
    data['flow_compression'] = (data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1) + 1e-8) - 1
    
    data['flow_accumulation'] = ((data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)) * \
                               (data['volume'] / (data['volume'].shift(1) + 1e-8))
    
    data['low_flow_core'] = data['flow_compression'] * data['flow_accumulation'] * data['value_flow_ratio']
    
    # Transition Flow Signals
    data['flow_regime_shift'] = data['flow_vol_momentum'] * data['vol_flow_divergence']
    
    data['early_flow_momentum'] = ((data['close'] - data['close'].shift(1)) / (data['close'].shift(1) + 1e-8)) * \
                                 (data['volume'] / (data['volume'].shift(1) + 1e-8))
    
    data['transition_flow_core'] = data['flow_regime_shift'] * data['early_flow_momentum'] * data['micro_range_eff']
    
    # Adaptive Flow Integration
    # Regime-Specific Flow Transmission
    data['high_flow_transmission'] = data['flow_transmission_core'] * data['high_flow_core'] * data['daily_range_vol']
    data['low_flow_transmission'] = data['flow_transmission_core'] * data['low_flow_core'] * (1 / (data['daily_range_vol'] + 1e-8))
    data['transition_flow_transmission'] = data['flow_transmission_core'] * data['transition_flow_core'] * data['micro_range_eff']
    
    # Flow Persistence Metrics
    data['fractal_eff_sign'] = np.sign(data['fractal_eff_cascade'])
    data['vol_flow_cascade_sign'] = np.sign(data['vol_flow_cascade'])
    data['eff_persistence'] = data['fractal_eff_sign'].rolling(window=3, min_periods=3).apply(
        lambda x: (x == x.shift(1)).sum() / 3 if len(x) == 3 else np.nan)
    
    data['micro_range_eff_sign'] = np.sign(data['micro_range_eff'])
    data['flow_compression_sign'] = np.sign(data['flow_compression'])
    data['range_flow_persistence'] = data['micro_range_eff_sign'].rolling(window=3, min_periods=3).apply(
        lambda x: (x == data['flow_compression_sign'].loc[x.index]).sum() / 3 if len(x) == 3 else np.nan)
    
    # Flow regime persistence
    data['current_regime'] = np.where(data['high_flow_regime'], 1, 
                                     np.where(data['low_flow_regime'], -1, 0))
    data['flow_regime_persistence'] = data['current_regime'].rolling(window=5, min_periods=5).apply(
        lambda x: (x == x.shift(1)).sum() / 5 if len(x) == 5 else np.nan)
    
    # Dynamic Flow Weighting
    data['flow_vol_weight'] = data['daily_range_vol'] / (data['daily_range_vol'] + data['daily_range_vol'].shift(1) + 1e-8)
    data['flow_volume_weight'] = data['volume'] / (data['volume'] + data['volume'].shift(1) + 1e-8)
    data['flow_persistence_weight'] = data['eff_persistence'] * data['range_flow_persistence'] * data['flow_regime_persistence']
    
    # Final Alpha Synthesis
    # Core Signal Selection
    data['conditional_flow_core'] = np.where(data['high_flow_regime'], data['high_flow_transmission'],
                                            np.where(data['low_flow_regime'], data['low_flow_transmission'],
                                                    data['transition_flow_transmission']))
    
    data['fractal_flow_enhancement'] = data['conditional_flow_core'] * data['fractal_eff_cascade'] * data['vol_flow_cascade']
    data['range_flow_enhancement'] = data['fractal_flow_enhancement'] * data['micro_range_eff'] * data['flow_compression']
    
    # Volume Flow Confirmation
    data['flow_alignment_confirmation'] = data['range_flow_enhancement'] * data['flow_direction_alignment']
    data['value_flow_enhancement'] = data['flow_alignment_confirmation'] * data['value_flow_ratio']
    
    # Adaptive Flow Adjustment
    data['flow_vol_adjusted'] = data['value_flow_enhancement'] / (data['daily_range_vol'] + 1e-8)
    data['flow_persistence_adjusted'] = data['flow_vol_adjusted'] * data['flow_persistence_weight']
    data['flow_regime_adjusted'] = data['flow_persistence_adjusted'] * data['flow_vol_weight']
    
    # Final Flow Efficiency Alpha
    data['base_flow_alpha'] = data['flow_regime_adjusted'] * np.sign(data['flow_vol_momentum'])
    
    # Enhanced Flow Alpha
    data['base_flow_alpha_sign'] = np.sign(data['base_flow_alpha'])
    data['efficiency_momentum_sign'] = np.sign(data['efficiency_momentum'])
    data['enhancement_ratio'] = data['base_flow_alpha_sign'].rolling(window=3, min_periods=3).apply(
        lambda x: (x == data['efficiency_momentum_sign'].loc[x.index]).sum() / 3 if len(x) == 3 else np.nan)
    
    data['enhanced_flow_alpha'] = data['base_flow_alpha'] * data['enhancement_ratio']
    
    return data['enhanced_flow_alpha']
