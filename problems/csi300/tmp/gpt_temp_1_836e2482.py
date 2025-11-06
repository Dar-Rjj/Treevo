import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Volatility Regime Detection
    data['volatility_ratio'] = (data['high'] - data['low']) / (data['high'].shift(5) - data['low'].shift(5))
    data['regime'] = 'normal'
    data.loc[data['volatility_ratio'] > 1.2, 'regime'] = 'high'
    data.loc[data['volatility_ratio'] < 0.8, 'regime'] = 'low'
    
    # Multi-Timeframe Momentum Analysis
    data['short_term_return'] = data['close'] / data['close'].shift(1) - 1
    data['medium_term_return'] = data['close'] / data['close'].shift(5) - 1
    data['momentum_divergence'] = data['short_term_return'] / data['medium_term_return']
    data['reversal_signal'] = (np.sign(data['short_term_return']) != np.sign(data['medium_term_return'])).astype(int)
    
    # Volume-Weighted Price Anchoring
    data['volume_weighted_high'] = data['high'] * (data['volume'] / data['volume'].shift(1))
    data['volume_weighted_low'] = data['low'] * (data['volume'].shift(1) / data['volume'])
    data['anchor_spread'] = data['volume_weighted_high'] - data['volume_weighted_low']
    
    data['high_deviation'] = (data['close'] - data['volume_weighted_high']) / data['anchor_spread']
    data['low_deviation'] = (data['close'] - data['volume_weighted_low']) / data['anchor_spread']
    data['anchor_bias'] = data['high_deviation'] * data['low_deviation']
    
    # Anchor Persistence
    data['anchor_sign'] = np.sign(data['anchor_bias'])
    data['anchor_persistence'] = 0
    for i in range(1, len(data)):
        if data['anchor_sign'].iloc[i] == data['anchor_sign'].iloc[i-1]:
            data['anchor_persistence'].iloc[i] = data['anchor_persistence'].iloc[i-1] + 1
    
    # Volume-Pressure Dynamics
    data['volume_acceleration'] = data['volume'].rolling(5).sum() / data['volume'].shift(5).rolling(5).sum()
    
    data['up_pressure'] = ((data['close'] > data['open']).astype(int) * data['volume'] * (data['high'] - data['open']))
    data['down_pressure'] = ((data['close'] < data['open']).astype(int) * data['volume'] * (data['open'] - data['low']))
    data['net_pressure'] = data['up_pressure'] - data['down_pressure']
    
    # Direction Streak
    data['pressure_sign'] = np.sign(data['net_pressure'])
    data['direction_streak'] = 0
    for i in range(1, len(data)):
        if data['pressure_sign'].iloc[i] == data['pressure_sign'].iloc[i-1]:
            data['direction_streak'].iloc[i] = data['direction_streak'].iloc[i-1] + 1
    
    data['flow_strength'] = data['direction_streak'] * data['net_pressure']
    data['volume_efficiency'] = abs(data['close'] - data['open']) * data['volume'] / (data['high'] - data['low'])
    
    # Intraday Extremes Integration
    data['daily_strength'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['high_to_close_ratio'] = data['high'] / data['close']
    data['low_to_close_ratio'] = data['low'] / data['close']
    data['extremes_composite'] = (data['high_to_close_ratio'] + data['low_to_close_ratio']) * data['daily_strength']
    
    # Divergence Detection System
    data['momentum_quality'] = data['momentum_divergence'] * abs(data['short_term_return'])
    data['anchor_momentum_divergence'] = np.sign(data['momentum_quality'] * data['anchor_bias']) * abs(data['momentum_quality'] * data['anchor_bias'])
    data['reversal_enhanced_divergence'] = data['reversal_signal'] * data['anchor_momentum_divergence']
    
    data['volume_momentum_alignment'] = np.sign(data['momentum_quality'] * data['flow_strength']) * abs(data['momentum_quality'] * data['flow_strength'])
    data['pressure_momentum_alignment'] = np.sign(data['net_pressure'] * data['momentum_divergence']) * abs(data['net_pressure'] * data['momentum_divergence'])
    
    data['volatility_momentum'] = np.sign(data['volatility_ratio'] - 1) - np.sign(data['momentum_divergence'])
    
    # Gap Analysis and Absorption
    data['opening_gap'] = data['open'] / data['close'].shift(1) - 1
    data['absorption_rate'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['gap_quality'] = abs(data['absorption_rate']) * data['volume_acceleration']
    
    # Core Divergence Assembly
    data['base_divergence'] = data['reversal_enhanced_divergence'] * data['volume_momentum_alignment']
    data['anchor_enhanced'] = data['base_divergence'] * data['anchor_persistence']
    data['extremes_weighted'] = data['anchor_enhanced'] * data['extremes_composite']
    
    # Volume Confirmation Framework
    data['volume_efficiency_confirmation'] = data['extremes_weighted'] * data['volume_efficiency']
    data['flow_strength_filter'] = data['volume_efficiency_confirmation'] * data['flow_strength']
    data['acceleration_multiplier'] = data['flow_strength_filter'] * data['volume_acceleration']
    
    # Directional Signal Construction
    data['bullish_bias'] = (data['momentum_divergence'] > 1).astype(int) * data['acceleration_multiplier'] * data['volume_efficiency']
    data['bearish_bias'] = (data['momentum_divergence'] < 1).astype(int) * data['acceleration_multiplier'] * data['anchor_bias']
    data['directional_core'] = data['bullish_bias'] + data['bearish_bias']
    
    # Final Alpha Generation
    data['persistence_enhancement'] = np.maximum(data['anchor_persistence'], data['direction_streak'])
    data['quality_adjustment'] = data['momentum_quality'] * data['volume_efficiency']
    
    # Regime-Adaptive Scaling
    data['regime_scaled_signal'] = 0
    high_vol_mask = data['regime'] == 'high'
    low_vol_mask = data['regime'] == 'low'
    normal_vol_mask = data['regime'] == 'normal'
    
    data.loc[high_vol_mask, 'regime_scaled_signal'] = data.loc[high_vol_mask, 'directional_core'] * data.loc[high_vol_mask, 'persistence_enhancement']
    data.loc[low_vol_mask, 'regime_scaled_signal'] = data.loc[low_vol_mask, 'directional_core'] * data.loc[low_vol_mask, 'quality_adjustment']
    data.loc[normal_vol_mask, 'regime_scaled_signal'] = data.loc[normal_vol_mask, 'directional_core'] * (data.loc[normal_vol_mask, 'persistence_enhancement'] + data.loc[normal_vol_mask, 'quality_adjustment']) / 2
    
    # Final Alpha with Daily Strength
    data['final_alpha'] = data['regime_scaled_signal'] * data['daily_strength']
    
    return data['final_alpha']
