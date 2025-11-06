import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Intraday Volatility Components
    data['true_range_vol'] = (data['high'] - data['low']) / data['close']
    data['gap_vol'] = abs(data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['efficiency_vol'] = (data['high'] - data['low']) / abs(data['close'] - data['open'])
    data['efficiency_vol'] = np.where(data['efficiency_vol'] == np.inf, 1, data['efficiency_vol'])
    data['efficiency_vol'] = np.where(data['efficiency_vol'] == -np.inf, 1, data['efficiency_vol'])
    
    # Rolling Volatility Regimes
    data['short_term_vol'] = data['true_range_vol'].rolling(window=3, min_periods=1).sum()
    data['medium_term_vol'] = data['true_range_vol'].rolling(window=6, min_periods=1).sum()
    data['long_term_vol'] = data['true_range_vol'].rolling(window=11, min_periods=1).sum()
    
    # Volatility Regime Classification
    data['high_vol_regime'] = data['short_term_vol'] > (data['medium_term_vol'] * 1.2)
    data['low_vol_regime'] = data['short_term_vol'] < (data['medium_term_vol'] * 0.8)
    data['medium_vol_regime'] = ~(data['high_vol_regime'] | data['low_vol_regime'])
    
    # Price Acceleration Analysis
    data['short_term_momentum'] = data['close'] / data['close'].shift(5) - 1
    data['acceleration'] = data['short_term_momentum'] / data['short_term_momentum'].shift(3) - 1
    
    # Volume Asymmetry Analysis
    up_days = data['close'] > data['close'].shift(1)
    down_days = data['close'] < data['close'].shift(1)
    
    up_volume_sum = data['volume'].rolling(window=5).apply(
        lambda x: np.sum(x * (data.loc[x.index, 'close'] > data.loc[x.index, 'close'].shift(1))), 
        raw=False
    )
    down_volume_sum = data['volume'].rolling(window=5).apply(
        lambda x: np.sum(x * (data.loc[x.index, 'close'] < data.loc[x.index, 'close'].shift(1))), 
        raw=False
    )
    data['volume_asymmetry_ratio'] = up_volume_sum / (down_volume_sum + 1e-8)
    
    # Volume Flow Patterns
    data['up_volume_intensity'] = (data['volume'] * np.maximum(0, data['close'] - data['open'])) / (data['high'] - data['low'] + 1e-8)
    data['down_volume_intensity'] = (data['volume'] * np.maximum(0, data['open'] - data['close'])) / (data['high'] - data['low'] + 1e-8)
    data['net_volume_flow'] = data['up_volume_intensity'] - data['down_volume_intensity']
    
    # Volume Flow Momentum
    data['short_term_vfm'] = data['net_volume_flow'] / (data['net_volume_flow'].shift(1) + 1e-8) - 1
    data['medium_term_vfm'] = data['net_volume_flow'] / (data['net_volume_flow'].rolling(window=3, min_periods=1).mean() + 1e-8) - 1
    data['volume_flow_persistence'] = np.sign(data['net_volume_flow']) * np.sign(data['net_volume_flow'].shift(1))
    
    # Regime Transition Detection
    data['volatility_breakout'] = data['short_term_vol'] / (data['medium_term_vol'] + 1e-8) - 1
    data['acceleration_shift'] = data['acceleration'] - data['acceleration'].shift(1)
    data['transition_signal'] = data['volatility_breakout'] * data['acceleration_shift']
    
    # Signal Quality Assessment
    data['volatility_consistency'] = data['short_term_vol'] / (data['medium_term_vol'] + 1e-8)
    data['volume_stability_score'] = data['volume'] / (data['volume'].rolling(window=5, min_periods=1).mean() + 1e-8)
    data['momentum_consistency'] = np.sign(data['short_term_momentum']) * np.sign(data['acceleration'])
    
    # High Volatility Regime Components
    volatility_normalized_acceleration = data['acceleration'] / (data['true_range_vol'] + 1e-8)
    volume_asymmetry_confirmation = data['volume_asymmetry_ratio'] - 1
    efficiency_multiplier = 1 / (data['efficiency_vol'] + 1e-8)
    
    high_vol_core_divergence = volatility_normalized_acceleration * volume_asymmetry_confirmation
    high_vol_volume_flow_adj = high_vol_core_divergence * data['volume_flow_persistence']
    gap_reaction_factor = data['gap_vol'] * np.sign(data['open'] - data['close'].shift(1))
    high_vol_base_signal = high_vol_volume_flow_adj * gap_reaction_factor * efficiency_multiplier
    
    # Low Volatility Regime Components
    raw_momentum = (data['close'] - data['close'].shift(5)) / (data['close'].shift(5) + 1e-8)
    acceleration_factor = 1 + data['acceleration']
    volatility_dampening = 1 / (data['long_term_vol'] + 1e-8)
    efficiency_stability = 1 / (abs(data['efficiency_vol'] - 1) + 1e-8)
    
    low_vol_base_signal = (raw_momentum * acceleration_factor * volatility_dampening * 
                          data['volume_asymmetry_ratio'] * data['medium_term_vfm'] * efficiency_stability)
    
    # Medium Volatility Regime (use weighted average)
    medium_vol_base_signal = (high_vol_base_signal + low_vol_base_signal) / 2
    
    # Regime-Specific Signal Selection
    base_signal = np.where(data['high_vol_regime'], high_vol_base_signal,
                          np.where(data['low_vol_regime'], low_vol_base_signal, medium_vol_base_signal))
    
    # Apply transition signal
    base_signal = base_signal * (1 + data['transition_signal'])
    
    # Volatility-Weighting
    volatility_weighted_signal = np.where(data['high_vol_regime'], base_signal / (data['true_range_vol'] + 1e-8),
                                         np.where(data['low_vol_regime'], base_signal * (1 / (data['long_term_vol'] + 1e-8)),
                                                 base_signal / (data['medium_term_vol'] + 1e-8)))
    
    # Signal Refinement
    quality_adjusted = volatility_weighted_signal * data['volume_stability_score']
    consistency_filtered = quality_adjusted * data['momentum_consistency']
    volatility_smoothed = consistency_filtered / (data['volatility_consistency'] + 1e-8)
    
    # Final Output Processing
    composite_factor = volatility_smoothed
    trend_confirmation = composite_factor * np.sign(composite_factor.shift(1))
    cross_sectional_signal = composite_factor - composite_factor.rolling(window=5, min_periods=1).mean()
    
    final_signal = cross_sectional_signal * trend_confirmation
    
    return final_signal
