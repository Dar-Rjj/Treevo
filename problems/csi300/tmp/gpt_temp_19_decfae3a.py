import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Price-Volume Relationship Analysis
    data['price_change'] = data['close'] - data['close'].shift(1)
    data['volume_change'] = data['volume'] - data['volume'].shift(1)
    
    # Divergence Detection Framework
    data['pv_direction_div'] = np.sign(data['price_change']) * np.sign(data['volume_change'])
    data['amplitude_div'] = np.abs(data['price_change']) / (np.abs(data['volume_change']) + 1e-8)
    data['efficiency_div'] = (np.abs(data['close'] - data['open']) / (data['volume'] + 1e-8)) - \
                            (np.abs(data['close'].shift(1) - data['open'].shift(1)) / (data['volume'].shift(1) + 1e-8))
    
    # Multi-Timeframe Divergence Patterns
    data['short_term_div'] = data['pv_direction_div'] + data['pv_direction_div'].shift(1)
    data['medium_term_div'] = data['pv_direction_div'].rolling(window=3, min_periods=1).sum()
    data['divergence_acceleration'] = data['medium_term_div'] - data['short_term_div']
    
    # Divergence Confirmation Signals
    data['volume_weighted_div'] = data['pv_direction_div'] * data['volume']
    data['range_adjusted_div'] = data['amplitude_div'] * (data['high'] - data['low'])
    data['persistence_div'] = data['pv_direction_div'].rolling(window=3, min_periods=1).apply(
        lambda x: (x < 0).sum(), raw=True)
    
    # Liquidity Absorption Dynamics
    data['volume_absorption_ratio'] = data['volume'] / (data['high'] - data['low'] + 1e-8)
    data['price_absorption_efficiency'] = np.abs(data['close'] - data['open']) / (data['amount'] + 1e-8)
    data['range_absorption_depth'] = (data['high'] - data['low']) / (data['volume'] + 1e-8)
    
    # Absorption Pattern Recognition
    gap_condition = np.abs(data['open'] - data['close'].shift(1)) > 0.02 * data['close'].shift(1)
    data['early_session_absorption'] = np.where(gap_condition, 
                                               data['volume'] / (data['volume'].shift(1) + 1e-8), 0)
    
    close_condition = np.abs(data['close'] - data['open']) > 0.02 * data['open']
    data['late_session_absorption'] = np.where(close_condition, 
                                              data['volume'] / (data['volume'].shift(1) + 1e-8), 0)
    
    vol_absorption_avg = data['volume_absorption_ratio'].rolling(window=5, min_periods=1).mean()
    data['continuous_absorption'] = (data['volume_absorption_ratio'] > vol_absorption_avg.shift(1)).rolling(
        window=3, min_periods=1).sum()
    
    # Absorption Momentum
    data['absorption_strength'] = data['volume_absorption_ratio'] - data['volume_absorption_ratio'].shift(2)
    data['efficiency_momentum'] = data['price_absorption_efficiency'] - data['price_absorption_efficiency'].shift(2)
    data['depth_momentum'] = data['range_absorption_depth'] - data['range_absorption_depth'].shift(3)
    
    # Multi-Scale Market Microstructure
    data['intraday_micro'] = (data['high'] - data['low']) / (data['close'] + 1e-8)
    data['daily_micro'] = np.abs(data['close'] - data['close'].shift(1)) / (data['close'].shift(1) + 1e-8)
    data['multi_day_micro'] = data['close'].rolling(window=5, min_periods=1).std() / \
                             (data['close'].rolling(window=5, min_periods=1).mean() + 1e-8)
    
    # Scale Interaction Patterns
    data['scale_convergence'] = data['intraday_micro'] * data['daily_micro']
    data['scale_divergence'] = data['intraday_micro'] / (data['daily_micro'] + 1e-8)
    data['scale_persistence'] = (data['scale_convergence'] > 0).rolling(window=3, min_periods=1).sum()
    
    # Scale-Adaptive Signals
    data['high_freq_signal'] = data['intraday_micro'] * data['volume']
    data['low_freq_signal'] = data['multi_day_micro'] * data['amount']
    data['cross_scale_signal'] = data['scale_convergence'] * data['scale_persistence']
    
    # Volume-Price Efficiency Framework
    data['up_move_volume_eff'] = np.where(data['close'] > data['open'], data['volume'], 0)
    data['down_move_volume_eff'] = np.where(data['close'] < data['open'], data['volume'], 0)
    data['efficiency_bias'] = data['up_move_volume_eff'] - data['down_move_volume_eff']
    
    # Efficiency Regime Classification
    eff_avg = data['price_absorption_efficiency'].rolling(window=5, min_periods=1).mean()
    data['high_efficiency'] = data['price_absorption_efficiency'] > eff_avg.shift(1)
    data['low_efficiency'] = data['price_absorption_efficiency'] < eff_avg.shift(1)
    
    # Efficiency Momentum Patterns
    data['efficiency_trend'] = data['efficiency_bias'] - data['efficiency_bias'].shift(2)
    
    def regime_persistence_calc(series):
        if len(series) < 2:
            return 0
        return (series == series.shift(1)).sum()
    
    data['regime_persistence'] = data['high_efficiency'].rolling(window=3, min_periods=1).apply(
        regime_persistence_calc, raw=False)
    
    data['volume_confirmed_efficiency'] = data['efficiency_trend'] * (data['volume'] / (data['volume'].shift(1) + 1e-8) - 1)
    
    # Divergence-Absorption Convergence
    data['div_absorption_coherence'] = data['pv_direction_div'] * data['volume_absorption_ratio']
    data['scale_efficiency_alignment'] = data['scale_convergence'] * data['price_absorption_efficiency']
    data['volume_efficiency_sync'] = data['efficiency_bias'] * data['volume_absorption_ratio']
    
    # Convergence Strength Evaluation
    div_coherence_positive = data['div_absorption_coherence'] > 0
    scale_alignment_positive = data['scale_efficiency_alignment'] > 0
    volume_sync_positive = data['volume_efficiency_sync'] > 0
    
    div_coherence_increasing = data['div_absorption_coherence'] > data['div_absorption_coherence'].shift(1)
    scale_alignment_increasing = data['scale_efficiency_alignment'] > data['scale_efficiency_alignment'].shift(1)
    volume_sync_increasing = data['volume_efficiency_sync'] > data['volume_efficiency_sync'].shift(1)
    
    data['strong_alignment'] = div_coherence_positive & scale_alignment_positive & volume_sync_positive & \
                              div_coherence_increasing & scale_alignment_increasing & volume_sync_increasing
    data['moderate_alignment'] = ((div_coherence_positive & scale_alignment_positive) | 
                                 (div_coherence_positive & volume_sync_positive) | 
                                 (scale_alignment_positive & volume_sync_positive))
    
    # Convergence Dynamics
    def alignment_duration_calc(series):
        if len(series) == 0:
            return 0
        current = series.iloc[-1]
        duration = 0
        for i in range(len(series)-2, -1, -1):
            if series.iloc[i] == current:
                duration += 1
            else:
                break
        return duration
    
    data['alignment_duration'] = data['strong_alignment'].rolling(window=10, min_periods=1).apply(
        alignment_duration_calc, raw=False)
    
    data['alignment_stability'] = (data['strong_alignment'] == data['strong_alignment'].shift(1)).rolling(
        window=3, min_periods=1).mean()
    
    data['alignment_momentum'] = data['alignment_stability'] * data['regime_persistence']
    
    # Adaptive Divergence Alpha Engine
    # Core Divergence Components
    data['price_volume_momentum'] = data['medium_term_div'] * data['divergence_acceleration']
    data['absorption_momentum'] = data['absorption_strength'] * data['efficiency_momentum']
    data['scale_momentum'] = data['cross_scale_signal'] * data['scale_persistence']
    data['efficiency_momentum_component'] = data['efficiency_trend'] * data['volume_confirmed_efficiency']
    
    # Convergence-Adaptive Weighting
    data['strong_alignment_weight'] = 1 + data['alignment_stability'] * data['alignment_duration']
    data['moderate_alignment_weight'] = 1 + 0.5 * data['alignment_stability'] * data['alignment_duration']
    data['weak_alignment_weight'] = 1 - 0.4 * (1 - data['alignment_stability'])
    
    # Apply weights based on alignment strength
    data['convergence_weight'] = np.where(data['strong_alignment'], data['strong_alignment_weight'],
                                         np.where(data['moderate_alignment'], data['moderate_alignment_weight'],
                                                 data['weak_alignment_weight']))
    
    data['efficiency_adjusted_weight'] = data['convergence_weight'] * (data['regime_persistence'] / 3 + 0.67)
    
    # Final Alpha Integration
    data['base_alpha'] = data['price_volume_momentum'] * data['absorption_momentum'] * data['scale_momentum']
    data['enhanced_alpha'] = data['base_alpha'] * data['efficiency_adjusted_weight'] * data['efficiency_momentum_component']
    
    # High-Efficiency Override
    he_condition = data['high_efficiency'] & (data['volume'] > 2 * data['volume'].shift(1))
    data['he_alpha'] = np.where(he_condition,
                               data['base_alpha'] * (data['volume'] / (data['volume'].shift(1) + 1e-8)) * 
                               data['absorption_strength'] * data['efficiency_trend'],
                               data['enhanced_alpha'])
    
    # Signal Confidence Framework
    high_conf_condition = data['strong_alignment'] & (data['regime_persistence'] >= 2) & \
                         (data['medium_term_div'] > 0) & (data['divergence_acceleration'] > 0)
    
    medium_conf_condition = data['moderate_alignment'] & (data['regime_persistence'] >= 1) & \
                           (data['medium_term_div'].abs() > 0.5)
    
    # Apply confidence-based final alpha
    data['final_alpha'] = np.where(high_conf_condition, data['he_alpha'] * 1.2,
                                  np.where(medium_conf_condition, data['he_alpha'] * 1.0,
                                          data['he_alpha'] * 0.8))
    
    # Clean up and return
    alpha_series = data['final_alpha'].fillna(0)
    alpha_series = alpha_series.replace([np.inf, -np.inf], 0)
    
    return alpha_series
