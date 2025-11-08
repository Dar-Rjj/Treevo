import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Volatility & Volume Regime Classification
    # Volatility Ratio
    data['range'] = data['high'] - data['low']
    data['vol_ratio_short'] = data['range'].rolling(window=5).mean()
    data['vol_ratio_long'] = data['range'].rolling(window=20).mean()
    data['volatility_ratio'] = data['vol_ratio_short'] / data['vol_ratio_long']
    
    # Volume Activity Ratio
    data['volume_min_10'] = data['volume'].rolling(window=10, min_periods=1).min()
    data['volume_activity_ratio'] = data['volume'] / data['volume_min_10']
    
    # Volume Distribution Intensity
    def percentile_80(x):
        return np.percentile(x, 80) if len(x) > 0 else 1.0
    
    data['volume_percentile_80'] = data['volume'].rolling(window=20, min_periods=1).apply(percentile_80, raw=True)
    data['volume_distribution_intensity'] = data['volume'] / data['volume_percentile_80']
    
    # Regime Classification
    conditions = [
        (data['volatility_ratio'] > 1.2) & (data['volume_activity_ratio'] > 1.5),
        (data['volatility_ratio'] < 0.8) & (data['volume_activity_ratio'] < 0.7),
        (data['volatility_ratio'] >= 0.8) & (data['volatility_ratio'] <= 1.2) & 
        (data['volume_activity_ratio'] >= 0.7) & (data['volume_activity_ratio'] <= 1.5)
    ]
    choices = ['high_vol_high_vol', 'low_vol_low_vol', 'normal_vol_moderate_vol']
    data['regime'] = np.select(conditions, choices, default='mixed_regime')
    
    # Asymmetric Microstructure Patterns
    # Price Path Efficiency Analysis
    data['actual_vs_min_distance'] = data['range'] / (
        abs(data['open'] - data['close']) + 
        abs(data['high'] - np.maximum(data['open'], data['close'])) + 
        abs(data['low'] - np.minimum(data['open'], data['close']))
    )
    
    # Up/Down Volatility Skew
    data['up_down_vol_skew'] = (data['high'] - data['close']) / (data['close'] - data['low'])
    data['up_down_vol_skew'] = data['up_down_vol_skew'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Price Rejection Strength
    data['price_rejection_strength'] = abs(data['close'] - (data['high'] + data['low'])/2) / data['range']
    data['price_rejection_strength'] = data['price_rejection_strength'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Gap & Absorption Behavior
    data['gap_filling_momentum'] = np.sign(data['open'] - data['close'].shift(1)) * (data['close'] - np.minimum(data['open'], data['close'].shift(1)))
    data['gap_filling_momentum'] = data['gap_filling_momentum'].fillna(0)
    
    data['absorption_ratio'] = (data['volume'] * abs(data['close'] - data['open'])) / data['range']
    data['absorption_ratio'] = data['absorption_ratio'].replace([np.inf, -np.inf], 0).fillna(0)
    
    data['gap_asymmetry_efficiency'] = data['gap_filling_momentum'] * data['absorption_ratio']
    
    # Directional Consistency Patterns
    data['close_abs_change'] = abs(data['close'] - data['close'].shift(1))
    data['price_direction_consistency'] = (data['close'] - data['open']) / data['close_abs_change'].rolling(window=5).sum()
    data['price_direction_consistency'] = data['price_direction_consistency'].replace([np.inf, -np.inf], 0).fillna(0)
    
    data['volume_value_alignment'] = np.sign(data['close'] - data['close'].shift(1)) * data['volume'] / data['range']
    data['volume_value_alignment'] = data['volume_value_alignment'].replace([np.inf, -np.inf], 0).fillna(0)
    
    data['microstructure_quality_score'] = data['price_direction_consistency'] * data['volume_value_alignment']
    
    # Multi-Timeframe Asymmetry Convergence
    # Short-term Microstructure (1-3 days)
    data['recent_efficiency_trend'] = data['actual_vs_min_distance'] - data['actual_vs_min_distance'].shift(2)
    data['gap_momentum_persistence'] = data['gap_asymmetry_efficiency'].rolling(window=3).mean()
    data['volume_intensity_momentum'] = data['volume_distribution_intensity'] / data['volume_distribution_intensity'].shift(3)
    data['volume_intensity_momentum'] = data['volume_intensity_momentum'].replace([np.inf, -np.inf], 1).fillna(1)
    
    # Medium-term Pattern Development (5-10 days)
    data['weekly_microstructure_quality'] = data['microstructure_quality_score'] - data['microstructure_quality_score'].shift(5)
    data['efficiency_divergence'] = data['actual_vs_min_distance'] / data['actual_vs_min_distance'].shift(5)
    data['efficiency_divergence'] = data['efficiency_divergence'].replace([np.inf, -np.inf], 1).fillna(1)
    
    data['absorption_trend'] = data['absorption_ratio'] / data['absorption_ratio'].shift(5)
    data['absorption_trend'] = data['absorption_trend'].replace([np.inf, -np.inf], 1).fillna(1)
    
    # Cross-Timeframe Alignment
    data['efficiency_momentum_alignment'] = np.sign(data['recent_efficiency_trend']) * np.sign(data['weekly_microstructure_quality'])
    data['volume_absorption_convergence'] = data['volume_intensity_momentum'] * data['absorption_trend']
    data['multi_timeframe_convergence_score'] = data['efficiency_momentum_alignment'] * data['volume_absorption_convergence']
    
    # Regime-Specific Signal Processing
    regime_factors = []
    
    for idx, row in data.iterrows():
        regime = row['regime']
        
        if regime == 'high_vol_high_vol':
            absorption_enhanced_momentum = row['gap_asymmetry_efficiency'] * row['absorption_ratio']
            volume_intensified_divergence = row['up_down_vol_skew'] * row['volume_distribution_intensity']
            regime_factor = absorption_enhanced_momentum * volume_intensified_divergence * row['volatility_ratio']
            
        elif regime == 'low_vol_low_vol':
            efficiency_value_focus = row['actual_vs_min_distance'] * row['microstructure_quality_score']
            quiet_period_signals = row['price_rejection_strength'] * row['gap_filling_momentum']
            regime_factor = efficiency_value_focus * quiet_period_signals
            
        elif regime == 'normal_vol_moderate_vol':
            balanced_microstructure = row['actual_vs_min_distance'] * row['price_direction_consistency']
            traditional_alignment = row['price_direction_consistency'] * row['volume_value_alignment']
            regime_factor = balanced_microstructure * traditional_alignment
            
        else:  # mixed_regime
            cross_regime_divergence = row['up_down_vol_skew'] - row['price_rejection_strength']
            volume_regime_interaction = row['volume_activity_ratio'] * row['volume_distribution_intensity']
            regime_factor = cross_regime_divergence * volume_regime_interaction
        
        regime_factors.append(regime_factor)
    
    data['regime_adapted_base'] = regime_factors
    
    # Composite Alpha Construction
    data['microstructure_momentum'] = data['microstructure_quality_score'] - data['microstructure_quality_score'].shift(3)
    data['efficiency_trend'] = data['actual_vs_min_distance'] - data['actual_vs_min_distance'].shift(5)
    
    # Final Alpha Generation
    data['base_regime_factor'] = data['regime_adapted_base']
    data['convergence_enhancement'] = data['base_regime_factor'] * data['multi_timeframe_convergence_score']
    data['momentum_confirmation'] = data['convergence_enhancement'] * np.sign(data['microstructure_momentum']) * abs(data['efficiency_trend'])
    
    # Final alpha factor
    alpha = data['momentum_confirmation'].fillna(0)
    
    return alpha
