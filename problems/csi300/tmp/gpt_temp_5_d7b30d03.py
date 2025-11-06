import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Fractal Divergence with Rejection Asymmetry alpha factor
    """
    data = df.copy()
    
    # Price Fractal Components
    data['daily_range_fractal'] = (data['high'] - data['low']) / (abs(data['close'] - data['close'].shift(1)) + 1e-8)
    data['gap_absorption_fractal'] = abs(data['open'] - data['close'].shift(1)) / (data['high'] - data['low'] + 1e-8)
    
    # Range persistence
    range_5d_avg = (data['high'] - data['low']).rolling(window=5, min_periods=3).mean()
    data['range_persistence'] = ((data['high'] - data['low']) > range_5d_avg).rolling(window=5, min_periods=3).sum()
    
    # Volume Fractal Components - Hurst-like calculation
    volume_changes = abs(data['volume'] - data['volume'].shift(1))
    data['hurst_volume'] = volume_changes.rolling(window=20, min_periods=10).apply(
        lambda x: np.log(x.std() / x.mean()) / np.log(len(x)) if len(x) > 1 and x.mean() > 0 else 0
    )
    
    # Volume clustering
    volume_median = data['volume'].rolling(window=20, min_periods=10).median()
    above_median = data['volume'] > volume_median
    data['volume_clustering'] = above_median.rolling(window=5, min_periods=3).apply(
        lambda x: (x == x.iloc[0]).sum() if len(x) > 0 else 0
    )
    
    # Fractal Regime Classification
    short_term_fractal = data['daily_range_fractal'].rolling(window=5, min_periods=3).mean()
    medium_term_fractal = data['daily_range_fractal'].rolling(window=20, min_periods=10).mean()
    data['fractal_transition'] = abs(short_term_fractal / (medium_term_fractal + 1e-8) - 1)
    
    # Multi-Scale Rejection Asymmetry
    data['upper_rejection'] = (data['high'] - np.maximum(data['open'], data['close'])) / (data['high'] - data['low'] + 1e-8)
    data['lower_rejection'] = (np.minimum(data['open'], data['close']) - data['low']) / (data['high'] - data['low'] + 1e-8)
    data['net_rejection_asymmetry'] = data['upper_rejection'] - data['lower_rejection']
    
    # Temporal Asymmetry Patterns
    mid_point = (data['high'] + data['low']) / 2
    data['morning_asymmetry'] = abs(mid_point - data['open']) / (data['high'] - data['low'] + 1e-8)
    data['afternoon_asymmetry'] = abs(data['close'] - mid_point) / (data['high'] - data['low'] + 1e-8)
    data['directional_bias'] = np.sign(data['close'] - data['open']) * np.sign(mid_point - data['open'])
    
    # Fractal-Scaled Rejection
    data['rejection_3d'] = (data['high'] - data['close'].rolling(window=3, min_periods=2).max()) - \
                          (data['close'].rolling(window=3, min_periods=2).min() - data['low'])
    data['rejection_10d'] = (data['high'] - data['close'].rolling(window=10, min_periods=5).max()) - \
                           (data['close'].rolling(window=10, min_periods=5).min() - data['low'])
    data['fractal_weighted_rejection'] = data['net_rejection_asymmetry'] * data['fractal_transition']
    
    # Volume-Range Momentum Dynamics
    data['amount_flow_velocity'] = data['amount'] / (data['amount'].shift(1) + 1e-8) - 1
    data['volume_acceleration'] = (data['volume'] / (data['volume'].shift(3) + 1e-8)) ** (1/3) - 1
    data['amount_flow_persistence'] = np.sign(data['amount'] - data['amount'].shift(1)) * \
                                    np.sign(data['amount'].shift(1) - data['amount'].shift(2))
    
    # Range-Volume Efficiency
    true_range = np.maximum(data['high'] - data['low'], 
                           np.maximum(abs(data['high'] - data['close'].shift(1)), 
                                     abs(data['low'] - data['close'].shift(1))))
    data['high_fractal_efficiency'] = data['amount'] / (data['volume'] * true_range + 1e-8)
    data['low_fractal_efficiency'] = data['amount'] / (data['volume'] * (data['high'] - data['low']) + 1e-8)
    data['efficiency_premium'] = data['high_fractal_efficiency'] - data['low_fractal_efficiency']
    
    # Volume-Momentum Alignment
    volume_ma_3d = data['volume'].rolling(window=3, min_periods=2).mean()
    volume_ma_5d = data['volume'].rolling(window=5, min_periods=3).mean()
    data['volume_trend_direction'] = np.sign(volume_ma_3d - volume_ma_5d)
    data['range_momentum_alignment'] = np.sign(data['close'] - data['close'].shift(1)) * data['volume_trend_direction']
    data['volume_confirmation_strength'] = abs(data['amount_flow_velocity']) * data['range_momentum_alignment']
    
    # Fractal Divergence Detection
    data['dimension_ratio'] = data['daily_range_fractal'] / (data['hurst_volume'] + 1e-8)
    
    # Rolling correlation between price and volume fractal dimensions
    data['fractal_correlation'] = data['daily_range_fractal'].rolling(window=10, min_periods=5).corr(data['hurst_volume'])
    data['fractal_divergence_signal'] = data['dimension_ratio'] * data['range_persistence']
    
    # Composite Alpha Construction
    # Primary Signal Components
    primary_1 = data['fractal_divergence_signal'] * data['net_rejection_asymmetry']
    primary_2 = data['efficiency_premium'] * data['directional_bias']
    primary_3 = data['volume_confirmation_strength'] * data['amount_flow_persistence']
    
    # Fractal-Adaptive Weighting
    range_autocorr = (data['high'] - data['low']).rolling(window=5, min_periods=3).apply(
        lambda x: x.autocorr() if len(x) > 1 else 0
    )
    high_fractal_weight = data['range_persistence'] / 5
    low_fractal_weight = 1 - abs(range_autocorr)
    transition_weight = data['fractal_transition']
    
    # Signal Integration
    weighted_signal = (primary_1 * high_fractal_weight + 
                      primary_2 * low_fractal_weight + 
                      primary_3 * transition_weight)
    
    # Volume-Range Momentum alignment adjustment
    momentum_adjustment = data['range_momentum_alignment'] * data['volume_confirmation_strength']
    
    # Divergence-Asymmetry interaction terms
    divergence_asymmetry = data['fractal_divergence_signal'] * data['fractal_weighted_rejection']
    
    # Final Factor Output
    composite_signal = weighted_signal + 0.3 * momentum_adjustment + 0.2 * divergence_asymmetry
    
    # Apply non-linear transformation
    final_alpha = 1 / (1 + np.exp(-composite_signal))
    
    # Incorporate regime transition signals
    regime_multiplier = 1 + 0.5 * data['fractal_transition']
    final_alpha = final_alpha * regime_multiplier
    
    # Combine with Volume Confirmation Strength
    final_alpha = final_alpha * (1 + 0.1 * data['volume_confirmation_strength'])
    
    return final_alpha
