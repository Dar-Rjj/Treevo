import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Microstructure Efficiency Momentum
    data['range_efficiency'] = (data['high'] - data['low']) / (data['close'] - data['open']).replace(0, np.nan)
    data['range_efficiency_accel'] = data['range_efficiency'] - data['range_efficiency'].shift(1)
    
    data['volume_adj_range'] = (data['high'] - data['low']) * np.log(data['volume'].replace(0, 1))
    data['volume_adj_range_momentum'] = data['volume_adj_range'] - data['volume_adj_range'].shift(1)
    
    data['large_trade_impact'] = data['amount'] / (data['high'] - data['low']).replace(0, np.nan)
    data['large_trade_impact_momentum'] = data['large_trade_impact'] - data['large_trade_impact'].shift(1)
    
    # Asymmetric Pressure Acceleration
    data['upside_pressure'] = (data['close'] - data['low']) * data['volume'] / (data['high'] - data['low']).replace(0, np.nan)
    data['upside_pressure_momentum'] = data['upside_pressure'] - data['upside_pressure'].shift(1)
    
    data['downside_pressure'] = (data['high'] - data['close']) * data['volume'] / (data['high'] - data['low']).replace(0, np.nan)
    data['downside_pressure_momentum'] = data['downside_pressure'] - data['downside_pressure'].shift(1)
    
    data['net_pressure'] = ((data['close'] - data['low']) - (data['high'] - data['close'])) * data['volume'] / (data['high'] - data['low']).replace(0, np.nan)
    data['net_pressure_accel'] = data['net_pressure'] - data['net_pressure'].shift(1)
    
    # Multi-Scale Breakout Efficiency
    data['short_breakout'] = (data['close'] - data['close'].shift(2)) / (data['high'].rolling(3).max() - data['low'].rolling(3).min()).replace(0, np.nan)
    data['short_breakout_momentum'] = data['short_breakout'] - (data['close'].shift(1) - data['close'].shift(3)) / (data['high'].shift(1).rolling(3).max() - data['low'].shift(1).rolling(3).min()).replace(0, np.nan)
    
    data['medium_breakout'] = (data['close'] - data['close'].shift(5)) / (data['high'].rolling(6).max() - data['low'].rolling(6).min()).replace(0, np.nan)
    data['medium_breakout_momentum'] = data['medium_breakout'] - (data['close'].shift(1) - data['close'].shift(6)) / (data['high'].shift(1).rolling(6).max() - data['low'].shift(1).rolling(6).min()).replace(0, np.nan)
    
    # Volatility-Volume Alignment
    data['range_volume_corr'] = (data['high'] - data['low']) * data['volume'] / data['amount'].replace(0, np.nan)
    data['range_volume_momentum'] = data['range_volume_corr'] - data['range_volume_corr'].shift(1)
    
    data['multi_scale_volume'] = data['volume'] / data['volume'].rolling(5).mean()
    data['multi_scale_volume_align'] = data['multi_scale_volume'] - (data['volume'].shift(1) / data['volume'].shift(1).rolling(5).mean())
    
    data['volatility_regime'] = (data['high'] - data['low']) / (data['high'].rolling(5).max() - data['low'].rolling(5).min()).replace(0, np.nan)
    data['volatility_regime_align'] = data['volatility_regime'] - (data['high'].shift(1) - data['low'].shift(1)) / (data['high'].shift(1).rolling(5).max() - data['low'].shift(1).rolling(5).min()).replace(0, np.nan)
    
    # Anchored Momentum with Microstructure Quality
    data['volume_weighted_anchor'] = (data['high'] * data['volume'] + data['low'] * data['volume']) / (2 * data['volume'])
    data['opening_gap_anchor'] = data['open'] / data['close'].shift(1)
    data['intraday_pressure_anchor'] = (data['close'] - (data['high'] + data['low'])/2) / (data['high'] - data['low']).replace(0, np.nan)
    
    data['short_anchored_momentum'] = (data['close'] - data['volume_weighted_anchor']) / data['close']
    data['medium_anchored_momentum'] = (data['close'] - data['volume_weighted_anchor'].rolling(5).mean()) / data['close']
    
    # Calculate momentum decay rate using rolling correlation
    momentum_decay = []
    for i in range(len(data)):
        if i >= 7:
            short_window = data['short_anchored_momentum'].iloc[i-6:i+1]
            medium_window = data['medium_anchored_momentum'].iloc[i-6:i+1]
            if len(short_window) == len(medium_window) and len(short_window) > 1:
                corr = np.corrcoef(short_window, medium_window)[0,1]
                momentum_decay.append(corr if not np.isnan(corr) else 0)
            else:
                momentum_decay.append(0)
        else:
            momentum_decay.append(0)
    data['momentum_decay_rate'] = momentum_decay
    
    # Price Discontinuity Impact
    data['overnight_gap'] = abs(data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['gap_persistence'] = abs(data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['gap_efficiency'] = data['gap_persistence'] / (1 + data['overnight_gap'])
    
    # Volume Flow Asymmetry
    data['upward_pressure_volume'] = np.where(data['close'] > data['open'], data['volume'], 0)
    data['downward_pressure_volume'] = np.where(data['close'] < data['open'], data['volume'], 0)
    
    data['volume_dominance_3d'] = (data['upward_pressure_volume'].rolling(3).sum() / 
                                  data['downward_pressure_volume'].rolling(3).sum()).replace([np.inf, -np.inf], np.nan)
    data['volume_pressure_ratio'] = np.log(data['volume_dominance_3d'].replace(0, 1))
    
    # Volume Concentration Quality
    data['high_volume_concentration'] = data['volume'].rolling(5).max() / data['volume'].rolling(5).sum()
    
    # Volume persistence calculation
    volume_persistence = []
    for i in range(len(data)):
        if i >= 9:
            window = data['volume'].iloc[i-9:i+1]
            threshold = window.mean()
            current_volume = data['volume'].iloc[i]
            if i >= 4:
                recent_window = data['volume'].iloc[i-4:i+1]
                persistence = sum(recent_window > threshold)
                volume_persistence.append(persistence)
            else:
                volume_persistence.append(0)
        else:
            volume_persistence.append(0)
    data['volume_persistence'] = volume_persistence
    data['volume_quality_score'] = data['high_volume_concentration'] * data['volume_persistence']
    
    # Price-Volume Efficiency
    price_change_sign = np.sign(data['close'] - data['close'].shift(1))
    volume_change_sign = np.sign(data['volume'] - data['volume'].shift(1))
    
    same_direction = (price_change_sign == volume_change_sign).rolling(5).sum()
    opposite_direction = (price_change_sign != volume_change_sign).rolling(5).sum()
    data['co_movement_efficiency'] = same_direction / (same_direction + opposite_direction).replace(0, np.nan)
    
    # Microstructure Noise Filtering
    data['upper_shadow_rejection'] = (data['high'] - np.maximum(data['open'], data['close'])) / (data['high'] - data['low']).replace(0, np.nan)
    data['lower_shadow_rejection'] = (np.minimum(data['open'], data['close']) - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    data['net_rejection_pressure'] = data['upper_shadow_rejection'] - data['lower_shadow_rejection']
    data['noise_filter_score'] = 1 / (1 + abs(data['net_rejection_pressure']))
    
    # Adaptive Factor Integration Framework
    data['base_momentum'] = data['short_anchored_momentum'] * data['momentum_decay_rate']
    data['volume_enhanced_momentum'] = data['base_momentum'] * data['volume_pressure_ratio']
    
    # Efficiency-Persistence Scoring
    data['range_efficiency_persistence'] = data['range_efficiency_accel'] * (1 - abs(data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan))
    data['volume_alignment_persistence'] = data['multi_scale_volume_align'] * data['volume_adj_range_momentum']
    
    # Breakout efficiency using short-term breakout momentum
    data['breakout_efficiency_score'] = data['short_breakout_momentum'] * data['medium_breakout_momentum']
    
    # Regime-Adaptive Weighting
    volatility_5d = (data['high'] - data['low']).rolling(5).mean()
    volatility_regime = np.where(volatility_5d > volatility_5d.quantile(0.7), 'high',
                               np.where(volatility_5d < volatility_5d.quantile(0.3), 'low', 'normal'))
    
    data['high_vol_weight'] = 0.3 * data['noise_filter_score']
    data['low_vol_weight'] = 0.7 * data['volume_quality_score']
    data['normal_vol_weight'] = 0.5 * data['co_movement_efficiency']
    
    # Apply regime weights
    regime_weight = np.where(volatility_regime == 'high', data['high_vol_weight'],
                           np.where(volatility_regime == 'low', data['low_vol_weight'], data['normal_vol_weight']))
    
    # Final Alpha Generation
    data['raw_signal'] = data['volume_enhanced_momentum'] * data['volume_quality_score']
    data['regime_weighted_signal'] = data['raw_signal'] * regime_weight
    data['final_alpha'] = data['regime_weighted_signal'] * data['breakout_efficiency_score']
    
    return data['final_alpha']
