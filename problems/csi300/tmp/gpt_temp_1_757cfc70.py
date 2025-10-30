import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Multi-Timeframe Efficiency Momentum factor
    """
    data = df.copy()
    
    # Multi-Timeframe Volatility-Efficiency Calculation
    # Short-Term (3-day) Fractal Efficiency
    data['close_ret_3'] = data['close'] - data['close'].shift(3)
    data['abs_ret_sum_3'] = data['close'].diff().abs().rolling(window=3, min_periods=3).sum()
    data['fractal_eff_3'] = data['close_ret_3'] / data['abs_ret_sum_3']
    
    # Medium-Term (8-day) Fractal Efficiency
    data['close_ret_8'] = data['close'] - data['close'].shift(8)
    data['abs_ret_sum_8'] = data['close'].diff().abs().rolling(window=8, min_periods=8).sum()
    data['fractal_eff_8'] = data['close_ret_8'] / data['abs_ret_sum_8']
    
    # Long-Term (21-day) Fractal Efficiency
    data['close_ret_21'] = data['close'] - data['close'].shift(21)
    data['abs_ret_sum_21'] = data['close'].diff().abs().rolling(window=21, min_periods=21).sum()
    data['fractal_eff_21'] = data['close_ret_21'] / data['abs_ret_sum_21']
    
    # Volatility Regime Detection & Classification
    # Short-Term Volatility
    data['high_range_3'] = data['high'].rolling(window=3, min_periods=3).max()
    data['low_range_3'] = data['low'].rolling(window=3, min_periods=3).min()
    data['volatility_3'] = data['abs_ret_sum_3'] / (data['high_range_3'] - data['low_range_3'])
    
    # Medium-Term Volatility
    data['high_range_8'] = data['high'].rolling(window=8, min_periods=8).max()
    data['low_range_8'] = data['low'].rolling(window=8, min_periods=8).min()
    data['volatility_8'] = data['abs_ret_sum_8'] / (data['high_range_8'] - data['low_range_8'])
    
    # Volatility Ratio
    data['volatility_ratio'] = data['volatility_3'] / data['volatility_8']
    
    # Regime Classification via Volatility Persistence
    data['vol_8_ma_5'] = data['volatility_8'].rolling(window=5, min_periods=5).mean()
    data['vol_threshold_high'] = 1.2 * data['vol_8_ma_5']
    data['vol_threshold_low'] = 0.8 * data['vol_8_ma_5']
    
    # Regime classification
    data['vol_regime'] = 1  # Normal by default
    data.loc[data['volatility_8'] > data['vol_threshold_high'], 'vol_regime'] = 2  # High
    data.loc[data['volatility_8'] < data['vol_threshold_low'], 'vol_regime'] = 0  # Low
    
    # Regime Transition Detection
    data['vol_transition'] = 1  # Stable by default
    data.loc[data['volatility_ratio'] > 1.2, 'vol_transition'] = 2  # Acceleration
    data.loc[data['volatility_ratio'] < 0.8, 'vol_transition'] = 0  # Deceleration
    
    # Efficiency-Dispersion Analysis
    # Multi-Timeframe Efficiency Consistency
    data['eff_sign_3_8'] = np.sign(data['fractal_eff_3']) * np.sign(data['fractal_eff_8'])
    data['eff_sign_8_21'] = np.sign(data['fractal_eff_8']) * np.sign(data['fractal_eff_21'])
    data['cross_timeframe_consistency'] = data['eff_sign_3_8'] * data['eff_sign_8_21']
    
    # Range-Normalized Efficiency Persistence
    data['range_eff_3'] = data['close_ret_3'] / (data['high_range_3'] - data['low_range_3'])
    data['range_eff_8'] = data['close_ret_8'] / (data['high_range_8'] - data['low_range_8'])
    data['range_eff_21'] = data['close_ret_21'] / (data['high'].rolling(window=21, min_periods=21).max() - 
                                                   data['low'].rolling(window=21, min_periods=21).min())
    
    # Volume-Weighted Momentum Confirmation
    # Volume-Weighted Volatility Momentum
    data['volume_weighted_3'] = (data['volume'] * data['close'].diff()).rolling(window=3, min_periods=3).sum() / data['abs_ret_sum_3']
    data['volume_weighted_8'] = (data['volume'] * data['close'].diff()).rolling(window=8, min_periods=8).sum() / data['abs_ret_sum_8']
    data['volume_weighted_21'] = (data['volume'] * data['close'].diff()).rolling(window=21, min_periods=21).sum() / data['abs_ret_sum_21']
    
    # Volume-Efficiency Convergence
    data['volume_weighted_efficiency'] = data['volume_weighted_8'] * data['fractal_eff_8']
    data['range_efficiency_confirmation'] = data['range_eff_8'] * data['fractal_eff_8']
    data['multi_dimensional_alignment'] = data['volume_weighted_efficiency'] * data['range_efficiency_confirmation']
    
    # Regime-Adaptive Signal Processing
    # Volatility-Regime Weighting
    regime_weights = []
    for i, row in data.iterrows():
        if row['vol_regime'] == 2:  # High Volatility
            regime_weights.append(0.7)
        elif row['vol_regime'] == 0:  # Low Volatility
            regime_weights.append(0.6)
        else:  # Normal Volatility
            regime_weights.append(0.5)
    data['regime_weight'] = regime_weights
    
    # Timeframe Adaptation
    timeframe_signals = []
    for i, row in data.iterrows():
        if row['vol_regime'] == 2:  # High Volatility - focus on 3-8 day
            signal = 0.6 * row['fractal_eff_3'] + 0.4 * row['fractal_eff_8']
        elif row['vol_regime'] == 0:  # Low Volatility - focus on 8-21 day
            signal = 0.4 * row['fractal_eff_8'] + 0.6 * row['fractal_eff_21']
        else:  # Normal Volatility - balanced approach
            signal = 0.33 * row['fractal_eff_3'] + 0.34 * row['fractal_eff_8'] + 0.33 * row['fractal_eff_21']
        timeframe_signals.append(signal)
    data['timeframe_signal'] = timeframe_signals
    
    # Transition Signal Enhancement
    transition_factors = []
    for i, row in data.iterrows():
        if row['vol_transition'] == 2:  # Acceleration
            factor = 1.3
        elif row['vol_transition'] == 0:  # Deceleration
            factor = 1.1
        else:  # Stable
            factor = 1.0
        transition_factors.append(factor)
    data['transition_factor'] = transition_factors
    
    # Adaptive Alpha Construction
    # Core Efficiency Components
    data['vol_eff_core'] = data['fractal_eff_8'] * data['cross_timeframe_consistency']
    data['volume_momentum_multiplier'] = data['volume_weighted_efficiency'] * data['multi_dimensional_alignment']
    data['range_efficiency_modifier'] = data['range_efficiency_confirmation'] * data['range_eff_8']
    
    # Volume-Price Validation
    data['volume_flow_confirmation'] = data['volume_weighted_8'] * np.sign(data['fractal_eff_8'])
    data['price_level_support'] = data['range_eff_8'] * data['fractal_eff_8']
    data['multi_dimensional_validation'] = data['volume_flow_confirmation'] * data['price_level_support']
    
    # Final Alpha Construction
    alpha_values = []
    for i, row in data.iterrows():
        # Core signal components
        core_signal = (row['vol_eff_core'] + row['volume_momentum_multiplier'] + row['range_efficiency_modifier']) / 3
        
        # Apply regime weighting
        weighted_signal = core_signal * row['regime_weight']
        
        # Apply transition enhancement
        enhanced_signal = weighted_signal * row['transition_factor']
        
        # Incorporate validation
        final_signal = enhanced_signal * (1 + 0.2 * row['multi_dimensional_validation'])
        
        alpha_values.append(final_signal)
    
    data['alpha'] = alpha_values
    
    return data['alpha']
