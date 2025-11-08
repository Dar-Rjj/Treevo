import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Scale Gap Asymmetry Dynamics
    data['overnight_gap'] = data['open'] / data['close'].shift(1) - 1
    data['intraday_gap'] = data['close'] / data['open'] - 1
    data['gap_absorption'] = np.abs(data['close'] - data['open']) / np.maximum(np.abs(data['open'] - data['close'].shift(1)), 1e-8)
    data['gap_asymmetry'] = data['overnight_gap'] / np.where(data['intraday_gap'] != 0, data['intraday_gap'], 1e-8)
    
    # Volume Acceleration Hierarchy
    data['volume_velocity'] = data['volume'] / data['volume'].shift(1) - 1
    data['volume_acceleration'] = data['volume_velocity'] - data['volume_velocity'].shift(1)
    
    # Volume momentum persistence
    volume_acc_sign = np.sign(data['volume_acceleration'])
    data['volume_persistence'] = 0
    for i in range(1, len(data)):
        if volume_acc_sign.iloc[i] == volume_acc_sign.iloc[i-1] and not pd.isna(volume_acc_sign.iloc[i-1]):
            data.iloc[i, data.columns.get_loc('volume_persistence')] = data.iloc[i-1, data.columns.get_loc('volume_persistence')] + 1
        else:
            data.iloc[i, data.columns.get_loc('volume_persistence')] = 1
    
    data['volume_concentration'] = (data['amount'] / np.maximum(data['volume'], 1e-8)) * np.abs(data['close'] - data['open'])
    
    # Momentum Divergence Framework
    data['price_momentum_2d'] = data['close'] / data['close'].shift(2) - 1
    data['volume_momentum_2d'] = data['volume'] / data['volume'].shift(2) - 1
    data['short_term_divergence'] = data['price_momentum_2d'] - data['volume_momentum_2d']
    
    data['return_1d'] = data['close'] / data['close'].shift(1) - 1
    data['return_3d'] = data['close'] / data['close'].shift(3) - 1
    data['return_10d'] = data['close'] / data['close'].shift(10) - 1
    
    data['short_term_acc'] = np.where(data['return_1d'] != 0, data['return_3d'] / data['return_1d'], 0)
    data['medium_term_acc'] = np.where(data['return_3d'] != 0, data['return_10d'] / data['return_3d'], 0)
    data['acceleration_hierarchy'] = data['short_term_acc'] - data['medium_term_acc']
    
    # Momentum regime classification
    data['momentum_aligned'] = ((data['price_momentum_2d'] > 0) & (data['volume_momentum_2d'] > 0)) | ((data['price_momentum_2d'] < 0) & (data['volume_momentum_2d'] < 0))
    data['momentum_divergent'] = ((data['price_momentum_2d'] > 0) & (data['volume_momentum_2d'] < 0)) | ((data['price_momentum_2d'] < 0) & (data['volume_momentum_2d'] > 0))
    data['reversal_detected'] = np.sign(data['acceleration_hierarchy']) != np.sign(data['acceleration_hierarchy'].shift(1))
    
    # Range Efficiency and Volatility Adaptation
    data['opening_efficiency'] = np.abs(data['open'] - data['close'].shift(1)) / np.maximum(data['high'] - data['low'], 1e-8)
    data['actual_range'] = (data['high'] - data['low']) / data['open']
    
    max_possible_range = np.maximum(np.abs(data['high'] - data['close'].shift(1)), np.abs(data['low'] - data['close'].shift(1)))
    data['potential_efficiency'] = (data['high'] - data['low']) / np.maximum(max_possible_range, 1e-8)
    
    # Volatility regime detection
    data['intraday_vol'] = data['high'] - data['low']
    data['vol_momentum'] = data['intraday_vol'] / data['intraday_vol'].shift(1)
    
    vol_rolling_mean = data['intraday_vol'].rolling(window=20, min_periods=10).mean()
    vol_rolling_std = data['intraday_vol'].rolling(window=20, min_periods=10).std()
    
    data['vol_regime_factor'] = 1.0
    high_vol_threshold = vol_rolling_mean + vol_rolling_std
    low_vol_threshold = vol_rolling_mean - vol_rolling_std
    
    data.loc[data['intraday_vol'] > high_vol_threshold, 'vol_regime_factor'] = 0.6
    data.loc[data['intraday_vol'] < low_vol_threshold, 'vol_regime_factor'] = 1.4
    
    # Volatility transition alignment
    data['vol_regime_change'] = data['vol_regime_factor'] != data['vol_regime_factor'].shift(1)
    vol_momentum_sign = np.sign(data['vol_momentum'])
    data['vol_persistence'] = 0
    for i in range(1, len(data)):
        if vol_momentum_sign.iloc[i] == vol_momentum_sign.iloc[i-1] and not pd.isna(vol_momentum_sign.iloc[i-1]):
            data.iloc[i, data.columns.get_loc('vol_persistence')] = data.iloc[i-1, data.columns.get_loc('vol_persistence')] + 1
    
    # Adaptive Factor Synthesis
    # Core gap-volume foundation
    core_gap_volume = data['gap_asymmetry'] * data['volume_acceleration']
    core_gap_volume = core_gap_volume * (1 + 0.1 * data['volume_persistence'])
    core_gap_volume = core_gap_volume * data['gap_absorption']
    
    # Momentum divergence adjustment
    momentum_divergence = data['short_term_divergence'] * data['acceleration_hierarchy']
    
    # Momentum regime multiplier
    regime_multiplier = np.where(data['momentum_aligned'], 1.2, 
                                np.where(data['momentum_divergent'], 0.8, 1.0))
    momentum_divergence = momentum_divergence * regime_multiplier
    
    # Reversal dampener
    momentum_divergence = np.where(data['reversal_detected'], momentum_divergence * 0.7, momentum_divergence)
    
    # Efficiency optimization layer
    efficiency_optimization = data['opening_efficiency'] * data['potential_efficiency']
    efficiency_optimization = efficiency_optimization * data['volume_concentration']
    
    # Dollar volume intensity
    data['dollar_volume_intensity'] = (data['close'] * data['volume']) / np.maximum(data['volume'].shift(1), 1e-8)
    efficiency_optimization = efficiency_optimization * data['dollar_volume_intensity']
    
    # Component integration
    integrated_factor = core_gap_volume * momentum_divergence
    integrated_factor = integrated_factor * efficiency_optimization
    integrated_factor = integrated_factor * data['volume_concentration']
    
    # Final regime adaptation
    final_factor = integrated_factor * data['vol_regime_factor']
    
    # Volatility transition alignment
    vol_transition_boost = 1 + (0.1 * data['vol_persistence'])
    final_factor = final_factor * vol_transition_boost
    
    # Momentum persistence reinforcement
    momentum_persistence_boost = 1 + (0.05 * data['volume_persistence'])
    final_factor = final_factor * momentum_persistence_boost
    
    # Clean up and return
    final_factor = final_factor.replace([np.inf, -np.inf], np.nan)
    return final_factor
