import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Microstructure Momentum Divergence System
    Adaptive regime-based alpha factor combining microstructure signals
    """
    data = df.copy()
    
    # Basic calculations
    data['intraday_range'] = data['high'] - data['low']
    data['intraday_move'] = data['close'] - data['open']
    data['efficiency_ratio'] = np.where(data['intraday_range'] > 0, 
                                      abs(data['intraday_move']) / data['intraday_range'], 0)
    data['vwap'] = data['amount'] / data['volume']
    
    # Price-Volume Microstructure Asymmetry components
    data['volume_ratio'] = data['volume'] / data['volume'].shift(1)
    data['intraday_momentum_efficiency'] = (data['intraday_move'] / data['intraday_range']) * (data['volume_ratio'] - 1)
    
    upper_shadow = data['high'] - np.maximum(data['open'], data['close'])
    lower_shadow = np.minimum(data['open'], data['close']) - data['low']
    data['microstructure_rejection'] = (upper_shadow - lower_shadow) * np.sign(data['intraday_move'])
    
    close_vol = data['close'] * data['volume']
    open_vol = data['open'] * data['volume']
    high_vol = data['high'] * data['volume']
    low_vol = data['low'] * data['volume']
    data['vw_price_momentum'] = np.where((high_vol - low_vol) != 0, 
                                       (close_vol - open_vol) / (high_vol - low_vol), 0)
    
    # Multi-Timeframe Volume Distribution
    data['vwap_ratio'] = data['vwap'] / data['vwap'].shift(1)
    data['volume_concentration_div'] = data['vwap_ratio'] * np.sign(data['intraday_move'])
    
    vol_ratio_3 = data['volume'] / data['volume'].shift(3)
    vol_ratio_6 = data['volume'] / data['volume'].shift(6)
    price_ratio_3 = data['close'] / data['close'].shift(3)
    price_ratio_6 = data['close'] / data['close'].shift(6)
    data['volume_persistence_momentum'] = (vol_ratio_3 - vol_ratio_6) * (price_ratio_3 - price_ratio_6)
    
    volume_spike = data['volume'] > 1.8 * data['volume'].shift(1)
    data['volume_spike_asymmetry'] = np.where(volume_spike, 
                                            data['intraday_move'] / data['intraday_range'], 0)
    
    # Gap Microstructure Behavior
    gap_size = abs(data['open'] - data['close'].shift(1))
    significant_gap = gap_size > 0.01 * data['close'].shift(1)
    data['gap_absorption_momentum'] = np.where(significant_gap, 
                                             data['intraday_move'] / gap_size, 0)
    
    large_gap = gap_size > 0.015 * data['close'].shift(1)
    data['gap_volume_intensity'] = np.where(large_gap, data['volume_ratio'], 1)
    
    gap_down_recovery = (data['open'] > data['close'].shift(1)) & (data['close'] < data['open'])
    data['intraday_gap_recovery'] = np.where(gap_down_recovery, 
                                           (data['close'] - data['low']) / (data['high'] - data['open']), 0)
    
    # Momentum-Velocity Fractal Alignment
    price_vel_3 = data['close'] / data['close'].shift(3) - 1
    price_vel_10 = data['close'] / data['close'].shift(10) - 1
    vol_vel_3 = data['volume'] / data['volume'].shift(3) - 1
    data['velocity_acceleration_div'] = price_vel_3 - price_vel_10 * vol_vel_3
    
    price_sign = np.sign(data['close'] - data['close'].shift(3))
    volume_sign = np.sign(data['volume'] - data['volume'].shift(3))
    data['momentum_velocity_consistency'] = price_sign * volume_sign * abs(data['intraday_move'])
    
    price_change = abs(data['close'] - data['close'].shift(1))
    volume_change = abs(data['volume'] - data['volume'].shift(1))
    data['fractal_efficiency_div'] = (price_change / data['intraday_range']) * (volume_change / data['volume'])
    
    # Microstructure Velocity Patterns
    data['price_velocity_corr'] = (data['intraday_move'] * (data['volume'] - data['volume'].shift(1))) / data['intraday_range']
    data['vw_range_efficiency'] = abs(data['intraday_move']) * data['volume'] / data['intraday_range'] * data['amount']
    data['tick_velocity_momentum'] = (data['high'] - data['open']) * data['volume'] - (data['open'] - data['low']) * data['volume']
    
    # Range-Velocity Dynamics
    range_ratio = data['intraday_range'] / data['intraday_range'].shift(1)
    data['range_expansion_velocity'] = range_ratio * data['volume_ratio']
    data['intraday_velocity_efficiency'] = data['efficiency_ratio'] * data['vwap']
    
    vol_persistence = data['volume'] / data['volume'].shift(2) - 1
    price_persistence = data['close'] / data['close'].shift(2) - 1
    data['velocity_persistence'] = vol_persistence * price_persistence * np.sign(data['intraday_move'])
    
    # Regime classification
    high_velocity = (data['volume'] > 1.5 * data['volume'].shift(1)) & (abs(data['intraday_move']) > 0.015 * data['open'])
    low_velocity = (data['volume'] < 0.8 * data['volume'].shift(1)) & (data['intraday_range'] < 0.01 * data['open'])
    
    high_efficiency = (data['efficiency_ratio'] > 0.7) & (data['intraday_move'].rolling(3).apply(lambda x: len(set(np.sign(x))) == 1))
    low_efficiency = (data['efficiency_ratio'] < 0.3) & (data['intraday_move'].rolling(3).apply(lambda x: len(set(np.sign(x))) > 1))
    
    # Adaptive signal blending based on regimes
    alpha_components = pd.DataFrame(index=data.index)
    
    # High Velocity Concentration Regime
    high_vel_signal = (data['velocity_acceleration_div'] * data['volume_concentration_div'] +
                      data['microstructure_rejection'] * data['tick_velocity_momentum'] +
                      data['intraday_velocity_efficiency'] * data['range_expansion_velocity'] +
                      data['price_velocity_corr'] * data['vw_price_momentum'])
    alpha_components['high_velocity'] = high_vel_signal * high_velocity.astype(int)
    
    # Low Velocity Diffusion Regime
    low_vel_signal = (data['volume_persistence_momentum'] * data['fractal_efficiency_div'] +
                     -1 * data['momentum_velocity_consistency'] * data['gap_absorption_momentum'] +
                     data['intraday_gap_recovery'] * data['velocity_persistence'] +
                     data['vw_range_efficiency'] * data['volume_spike_asymmetry'])
    alpha_components['low_velocity'] = low_vel_signal * low_velocity.astype(int)
    
    # High Efficiency Trending Regime
    high_eff_signal = (data['momentum_velocity_consistency'] * data['range_expansion_velocity'] +
                      data['velocity_acceleration_div'] * data['vw_range_efficiency'] +
                      data['volume_concentration_div'] * data['intraday_momentum_efficiency'] +
                      data['price_velocity_corr'] * data['intraday_velocity_efficiency'])
    alpha_components['high_efficiency'] = high_eff_signal * high_efficiency.astype(int)
    
    # Low Efficiency Ranging Regime
    low_eff_signal = (data['volume_concentration_div'] * data['vw_range_efficiency'] +
                     data['range_expansion_velocity'] * data['volume_spike_asymmetry'] +
                     data['gap_absorption_momentum'] * data['velocity_persistence'] +
                     -1 * data['momentum_velocity_consistency'] * data['fractal_efficiency_div'])
    alpha_components['low_efficiency'] = low_eff_signal * low_efficiency.astype(int)
    
    # Transition regime (default when no specific regime detected)
    transition_signal = (data['intraday_momentum_efficiency'] + 
                        data['microstructure_rejection'] + 
                        data['vw_price_momentum'] + 
                        data['velocity_acceleration_div'])
    transition_mask = ~(high_velocity | low_velocity | high_efficiency | low_efficiency)
    alpha_components['transition'] = transition_signal * transition_mask.astype(int)
    
    # Composite alpha with regime weighting
    regime_weights = pd.DataFrame({
        'high_velocity': high_velocity.astype(int),
        'low_velocity': low_velocity.astype(int),
        'high_efficiency': high_efficiency.astype(int),
        'low_efficiency': low_efficiency.astype(int),
        'transition': transition_mask.astype(int)
    })
    
    # Normalize weights to sum to 1
    regime_weights = regime_weights.div(regime_weights.sum(axis=1), axis=0).fillna(0)
    
    # Final composite alpha
    composite_alpha = (alpha_components * regime_weights).sum(axis=1)
    
    # Apply multi-scale confirmation
    short_term_confirmation = data['price_velocity_corr'].rolling(3).mean()
    medium_term_confirmation = data['velocity_acceleration_div'].rolling(5).mean()
    long_term_confirmation = data['volume_persistence_momentum'].rolling(10).mean()
    
    final_alpha = (composite_alpha * 
                  (1 + 0.3 * short_term_confirmation) * 
                  (1 + 0.2 * medium_term_confirmation) * 
                  (1 + 0.1 * long_term_confirmation))
    
    return final_alpha
