import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Market State Detection - Price-Based Regime Classification
    # Daily Price Range Efficiency
    data['efficiency_ratio'] = np.abs((data['close'] - data['open']) / (data['high'] - data['low']))
    data['efficiency_ratio'] = data['efficiency_ratio'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Rolling Regime Indicators
    data['eff_ratio_mean_10'] = data['efficiency_ratio'].rolling(window=10, min_periods=5).mean()
    data['eff_ratio_std_10'] = data['efficiency_ratio'].rolling(window=10, min_periods=5).std()
    data['eff_ratio_zscore'] = (data['efficiency_ratio'] - data['eff_ratio_mean_10']) / data['eff_ratio_std_10']
    data['eff_ratio_zscore'] = data['eff_ratio_zscore'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Regime Assignment
    data['regime'] = 'normal'
    data.loc[data['efficiency_ratio'] > (data['eff_ratio_mean_10'] + 0.5 * data['eff_ratio_std_10']), 'regime'] = 'high'
    data.loc[data['efficiency_ratio'] < (data['eff_ratio_mean_10'] - 0.5 * data['eff_ratio_std_10']), 'regime'] = 'low'
    
    # Volume-Based Confirmation
    data['volume_median_20'] = data['volume'].rolling(window=20, min_periods=10).median()
    data['volume_ratio'] = data['volume'] / data['volume_median_20']
    data['volume_spike'] = data['volume_ratio'] > 2.0
    
    # Volume-Price Consistency
    data['price_sign'] = np.sign(data['close'] - data['open'])
    data['volume_sign'] = np.sign(data['volume_ratio'] - 1)
    data['sign_alignment'] = data['price_sign'] * data['volume_sign']
    data['strong_consistency'] = (data['sign_alignment'] > 0) & (data['volume_ratio'] > 1.5)
    data['weak_consistency'] = (data['sign_alignment'] < 0) | (data['volume_ratio'] < 1.0)
    
    # Multi-Timeframe Acceleration Signals
    # Ultra-Short Acceleration (3-day)
    data['price_change_3d'] = data['close'] - data['close'].shift(3)
    data['price_change_1d'] = data['close'] - data['close'].shift(1)
    data['price_accel_ultra'] = data['price_change_1d'] - (data['price_change_3d'] / 3)
    
    data['volume_sum_3d'] = data['volume'].rolling(window=3, min_periods=2).sum()
    data['volume_accel_ultra'] = data['volume'] - (data['volume_sum_3d'] / 3)
    
    # Short-Term Acceleration (8-day)
    data['price_change_8d'] = data['close'] - data['close'].shift(8)
    data['price_change_4d'] = data['close'] - data['close'].shift(4)
    data['price_accel_short'] = data['price_change_4d'] - (data['price_change_8d'] / 2)
    
    data['volume_median_8d'] = data['volume'].rolling(window=8, min_periods=5).median()
    data['volume_median_4d'] = data['volume'].rolling(window=4, min_periods=3).median()
    data['volume_accel_short'] = data['volume_median_4d'] - data['volume_median_8d']
    
    # Medium-Term Reversal Acceleration (21-day)
    data['price_change_21d'] = data['close'] - data['close'].shift(21)
    data['price_change_7d'] = data['close'] - data['close'].shift(7)
    data['price_accel_medium'] = data['price_change_7d'] - (data['price_change_21d'] / 3)
    
    # Volume percentile ranks for reversal
    data['volume_rank_21d'] = data['volume'].rolling(window=21, min_periods=15).apply(
        lambda x: (x.rank(pct=True).iloc[-1] if len(x) >= 15 else 0.5), raw=False
    )
    data['volume_rank_7d'] = data['volume'].rolling(window=7, min_periods=5).apply(
        lambda x: (x.rank(pct=True).iloc[-1] if len(x) >= 5 else 0.5), raw=False
    )
    data['volume_reversal_medium'] = data['volume_rank_7d'] - data['volume_rank_21d']
    
    # Cross-Timeframe Convergence Detection
    data['accel_sign_ultra'] = np.sign(data['price_accel_ultra'])
    data['accel_sign_short'] = np.sign(data['price_accel_short'])
    data['accel_sign_medium'] = np.sign(data['price_accel_medium'])
    
    data['alignment_score'] = (
        (data['accel_sign_ultra'] == data['accel_sign_short']).astype(int) +
        (data['accel_sign_ultra'] == data['accel_sign_medium']).astype(int) +
        (data['accel_sign_short'] == data['accel_sign_medium']).astype(int)
    )
    
    # Regime-Adaptive Signal Fusion
    data['regime_weight_ultra'] = 0.0
    data['regime_weight_short'] = 0.0
    data['regime_weight_medium'] = 0.0
    
    # High Efficiency Regime
    high_mask = data['regime'] == 'high'
    data.loc[high_mask, 'regime_weight_ultra'] = 0.6
    data.loc[high_mask, 'regime_weight_short'] = 0.3
    data.loc[high_mask, 'regime_weight_medium'] = 0.1
    
    # Low Efficiency Regime
    low_mask = data['regime'] == 'low'
    data.loc[low_mask, 'regime_weight_ultra'] = 0.1
    data.loc[low_mask, 'regime_weight_short'] = 0.3
    data.loc[low_mask, 'regime_weight_medium'] = 0.6
    
    # Normal Regime
    normal_mask = data['regime'] == 'normal'
    data.loc[normal_mask, 'regime_weight_ultra'] = 0.33
    data.loc[normal_mask, 'regime_weight_short'] = 0.33
    data.loc[normal_mask, 'regime_weight_medium'] = 0.34
    
    # Apply volume-price consistency weighting
    data['consistency_multiplier'] = 1.0
    data.loc[data['strong_consistency'], 'consistency_multiplier'] = 1.5
    data.loc[data['weak_consistency'], 'consistency_multiplier'] = 0.7
    
    # Efficiency scaling
    data['efficiency_scale'] = 1.0
    data.loc[high_mask, 'efficiency_scale'] = data.loc[high_mask, 'efficiency_ratio'] / data.loc[high_mask, 'eff_ratio_mean_10']
    data.loc[low_mask, 'efficiency_scale'] = data.loc[low_mask, 'eff_ratio_mean_10'] / data.loc[low_mask, 'efficiency_ratio']
    data['efficiency_scale'] = data['efficiency_scale'].clip(0.5, 2.0)
    
    # Final Alpha Construction
    # Signal Integration
    data['weighted_accel'] = (
        data['regime_weight_ultra'] * data['price_accel_ultra'] * data['volume_accel_ultra'] +
        data['regime_weight_short'] * data['price_accel_short'] * data['volume_accel_short'] +
        data['regime_weight_medium'] * data['price_accel_medium'] * data['volume_reversal_medium']
    )
    
    # Apply cross-timeframe alignment multiplier
    data['alignment_multiplier'] = 1.0 + (data['alignment_score'] * 0.2)
    
    # Integrated signal with adjustments
    data['integrated_signal'] = (
        data['weighted_accel'] * 
        data['consistency_multiplier'] * 
        data['alignment_multiplier'] * 
        data['efficiency_scale']
    )
    
    # Non-Linear Enhancement
    data['enhanced_signal'] = np.tanh(data['integrated_signal'])
    data['final_signal'] = np.sign(data['enhanced_signal']) * np.sqrt(np.abs(data['enhanced_signal']))
    
    # Regime-Contextual Output
    data['regime_scale'] = 1.0
    data.loc[high_mask, 'regime_scale'] = 1.2
    data.loc[low_mask, 'regime_scale'] = 0.8
    
    # Volume spike adjustment
    data['volume_spike_adj'] = 1.0
    data.loc[data['volume_spike'], 'volume_spike_adj'] = 1.3
    
    # Final acceleration factor
    data['acceleration_factor'] = data['final_signal'] * data['regime_scale'] * data['volume_spike_adj']
    
    # Return the final factor series
    return data['acceleration_factor']
