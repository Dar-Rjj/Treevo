import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Add epsilon to avoid division by zero
    epsilon = 1e-8
    
    # Volatility Regime Classification
    data['prev_day_range'] = data['high'].shift(1) - data['low'].shift(1)
    data['prev_2day_range'] = data['high'].shift(2) - data['low'].shift(2)
    
    # Intraday Volatility Structure
    data['opening_vol_regime'] = (data['high'] - data['low']) / (data['prev_day_range'] + epsilon)
    data['closing_vol_regime'] = (data['high'] - data['low']) / (data['prev_2day_range'] + epsilon)
    data['vol_regime_ratio'] = data['opening_vol_regime'] / (data['closing_vol_regime'] + epsilon)
    
    # Volume-Volatility Relationship
    data['volume_efficiency'] = data['volume'] / (data['high'] - data['low'] + epsilon)
    data['vol_adjusted_volume'] = data['volume'] * (data['high'] - data['low']) / (data['prev_day_range'] + epsilon)
    data['vol_vol_alignment'] = data['volume_efficiency'] * data['vol_adjusted_volume']
    
    # Regime Transition Detection
    data['vol_expansion_signal'] = data['vol_regime_ratio'] * data['vol_vol_alignment']
    data['regime_change_conf'] = np.sign(data['vol_expansion_signal']) * np.sign(data['volume_efficiency'])
    
    # Price Efficiency Patterns
    data['opening_price_eff'] = (data['open'] - data['close'].shift(1)) / (data['prev_day_range'] + epsilon)
    data['opening_range_capture'] = (data['high'] - data['open']) / (data['open'] - data['low'] + epsilon)
    data['opening_eff_score'] = data['opening_price_eff'] * data['opening_range_capture']
    
    data['closing_price_eff'] = (data['close'] - data['open']) / (data['high'] - data['low'] + epsilon)
    data['closing_range_util'] = (data['close'] - data['low']) / (data['high'] - data['close'] + epsilon)
    data['closing_eff_score'] = data['closing_price_eff'] * data['closing_range_util']
    
    # Intraday Efficiency Divergence
    data['efficiency_direction'] = np.sign(data['opening_eff_score']) * np.sign(data['closing_eff_score'])
    data['efficiency_gap'] = data['opening_eff_score'] - data['closing_eff_score']
    
    # Multi-Timeframe Volume Dynamics
    data['short_vol_momentum'] = data['volume'] / (data['volume'].shift(1) + epsilon) - 1
    data['medium_vol_momentum'] = data['volume'] / (data['volume'].shift(3) + epsilon) - 1
    data['vol_momentum_div'] = data['short_vol_momentum'] - data['medium_vol_momentum']
    
    data['early_session_vol'] = data['volume'] * (data['open'] - data['low']) / (data['high'] - data['low'] + epsilon)
    data['late_session_vol'] = data['volume'] * (data['high'] - data['close']) / (data['high'] - data['low'] + epsilon)
    data['vol_dist_ratio'] = data['early_session_vol'] / (data['late_session_vol'] + epsilon)
    
    data['vol_timing_signal'] = data['vol_momentum_div'] * data['vol_dist_ratio']
    data['vol_timing_conf'] = np.sign(data['vol_timing_signal']) * np.sign(data['efficiency_gap'])
    
    # Price-Level Anchoring Effects
    data['high_anchor_prox'] = (data['high'] - data['high'].shift(1)) / (data['prev_day_range'] + epsilon)
    data['low_anchor_prox'] = (data['low'] - data['low'].shift(1)) / (data['prev_day_range'] + epsilon)
    data['anchor_asymmetry'] = data['high_anchor_prox'] - data['low_anchor_prox']
    
    data['open_high_anchor'] = (data['high'] - data['open']) / (data['open'] - data['low'] + epsilon)
    data['open_low_anchor'] = (data['open'] - data['low']) / (data['high'] - data['open'] + epsilon)
    data['opening_anchor_balance'] = data['open_high_anchor'] * data['open_low_anchor']
    
    data['anchor_eff_signal'] = data['anchor_asymmetry'] * data['opening_anchor_balance']
    data['anchor_conf'] = np.sign(data['anchor_eff_signal']) * np.sign(data['efficiency_gap'])
    
    # Signal Robustness Framework
    # Volatility Direction Stability (rolling count of same sign)
    vol_sign = np.sign(data['vol_regime_ratio'])
    data['vol_direction_stability'] = vol_sign.rolling(window=4, min_periods=1).apply(
        lambda x: sum(x == x.iloc[-1]) if len(x) > 0 else 1, raw=False
    )
    
    data['vol_magnitude_trend'] = data['vol_regime_ratio'].rolling(window=4, min_periods=1).mean()
    data['vol_consistency_score'] = data['vol_direction_stability'] * data['vol_magnitude_trend']
    
    # Efficiency Pattern Stability
    eff_sign = np.sign(data['efficiency_gap'])
    data['eff_direction_persistence'] = eff_sign.rolling(window=4, min_periods=1).apply(
        lambda x: sum(x == x.iloc[-1]) if len(x) > 0 else 1, raw=False
    )
    
    data['eff_magnitude_consistency'] = data['efficiency_gap'].abs().rolling(window=4, min_periods=1).mean()
    data['eff_stability_score'] = data['eff_direction_persistence'] * data['eff_magnitude_consistency']
    
    # Composite Robustness Framework
    data['signal_durability'] = data['vol_consistency_score'] * data['eff_stability_score']
    data['robust_enhanced_vol'] = data['vol_expansion_signal'] * data['signal_durability']
    data['robust_enhanced_eff'] = data['efficiency_gap'] * data['signal_durability']
    
    # Final Alpha Construction
    data['vol_eff_core'] = data['robust_enhanced_vol'] * data['robust_enhanced_eff']
    data['vol_timing_core'] = data['vol_timing_signal'] * data['vol_timing_conf']
    data['anchoring_core'] = data['anchor_eff_signal'] * data['anchor_conf']
    
    data['primary_alpha'] = data['vol_eff_core'] * data['vol_timing_core']
    data['secondary_enhance'] = data['primary_alpha'] * data['anchoring_core']
    data['robustness_adjust'] = data['secondary_enhance'] * data['signal_durability']
    
    data['alpha_core'] = data['robustness_adjust'] * data['vol_dist_ratio']
    data['final_alpha'] = data['alpha_core'] * data['efficiency_direction']
    
    return data['final_alpha']
