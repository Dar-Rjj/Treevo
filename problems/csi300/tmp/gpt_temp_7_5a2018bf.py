import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Small epsilon to avoid division by zero
    eps = 1e-8
    
    # Multi-Scale Volatility Regime Momentum
    # Micro Momentum
    data['micro_momentum'] = (data['close'] - data['open']) / (data['high'] - data['low'] + eps)
    
    # Meso Momentum
    data['high_3d'] = data['high'].rolling(window=3, min_periods=1).max()
    data['low_3d'] = data['low'].rolling(window=3, min_periods=1).min()
    data['meso_momentum'] = (data['close'] - data['open']) / (data['high_3d'] - data['low_3d'] + eps)
    
    # Momentum Divergence
    data['momentum_divergence'] = data['micro_momentum'] - data['meso_momentum']
    
    # Volatility Regime Classification
    data['opening_vol_regime'] = (data['high'] - data['low']) / (data['open'] + eps)
    data['closing_vol_regime'] = (data['high'] - data['low']) / (data['close'] + eps)
    data['vol_regime_divergence'] = data['opening_vol_regime'] - data['closing_vol_regime']
    
    # Regime-Adapted Momentum
    data['regime_adapted_momentum'] = data['momentum_divergence'] * data['vol_regime_divergence']
    
    # Volume-Fractal Integration
    # Volume Weighted Momentum
    data['volume_weighted_momentum'] = data['volume'] * data['regime_adapted_momentum']
    
    # Volume Persistence
    data['vwm_sign'] = np.sign(data['volume_weighted_momentum'])
    data['volume_persistence'] = data['vwm_sign'].groupby(data['vwm_sign'].ne(data['vwm_sign'].shift()).cumsum()).cumcount() + 1
    
    # Volume-Momentum Core
    data['volume_momentum_core'] = data['volume_weighted_momentum'] * data['volume_persistence']
    
    # Volume Flow Microstructure
    data['morning_volume_intensity'] = data['volume'] * (data['open'] - data['low']) / (data['high'] - data['low'] + eps)
    data['afternoon_volume_intensity'] = data['volume'] * (data['high'] - data['close']) / (data['high'] - data['low'] + eps)
    data['volume_distribution_asymmetry'] = data['morning_volume_intensity'] - data['afternoon_volume_intensity']
    
    # Fractal Volume Core
    data['fractal_volume_core'] = data['volume_momentum_core'] * data['volume_distribution_asymmetry']
    
    # Multi-Timeframe Regime Exhaustion
    # Fractal Exhaustion
    data['range_asymmetry'] = (data['high'] - data['open']) - (data['close'] - data['low'])
    data['volume_5d_avg'] = data['volume'].rolling(window=5, min_periods=1).mean().shift(1)
    data['volume_spike'] = data['volume'] / (data['volume_5d_avg'] + eps)
    data['exhaustion_signal'] = data['range_asymmetry'] * data['volume_spike']
    
    # Regime-Exhaustion Integration
    data['vol_range_5d'] = (data['high'] - data['low']).rolling(window=5, min_periods=1).mean()
    data['volatility_breakout_signal'] = (data['high'] - data['low']) / (data['vol_range_5d'] + eps)
    data['vol_exhaustion_alignment'] = np.sign(data['volatility_breakout_signal']) * np.sign(data['exhaustion_signal'])
    data['regime_weighted_exhaustion'] = data['exhaustion_signal'] * data['volatility_breakout_signal']
    
    # Multi-Timeframe Exhaustion Core
    data['multi_timeframe_exhaustion_core'] = data['regime_weighted_exhaustion'] * data['vol_exhaustion_alignment']
    
    # Microstructure Quality Framework
    # Pattern Consistency Metrics
    data['vol_regime_sign'] = np.sign(data['vol_regime_divergence'])
    data['regime_consistency'] = data['vol_regime_sign'].rolling(window=5, min_periods=1).apply(
        lambda x: (x == x.iloc[0]).sum() if len(x) > 0 else 1
    )
    
    data['momentum_sign'] = np.sign(data['regime_adapted_momentum'])
    data['momentum_consistency'] = data['momentum_sign'].rolling(window=5, min_periods=1).apply(
        lambda x: (x == x.iloc[0]).sum() if len(x) > 0 else 1
    )
    
    data['microstructure_quality_score'] = data['regime_consistency'] * data['momentum_consistency']
    
    # Volume Quality Metrics
    data['volume_asymmetry_sign'] = np.sign(data['volume_distribution_asymmetry'])
    data['volume_flow_consistency'] = data['volume_asymmetry_sign'].rolling(window=5, min_periods=1).apply(
        lambda x: (x == x.iloc[0]).sum() if len(x) > 0 else 1
    )
    
    def count_sign_changes(series):
        if len(series) < 2:
            return 0
        return (series != series.shift()).sum() - 1
    
    data['volume_sign_changes'] = data['volume_asymmetry_sign'].rolling(window=5, min_periods=1).apply(
        count_sign_changes, raw=False
    )
    data['volume_pattern_stability'] = data['volume_flow_consistency'] / (data['volume_sign_changes'] + 1)
    data['volume_quality_score'] = data['volume_flow_consistency'] * data['volume_pattern_stability']
    
    # Composite Quality Assessment
    data['regime_quality_weight'] = data['microstructure_quality_score'] * data['volume_quality_score']
    data['quality_enhanced_momentum'] = data['regime_adapted_momentum'] * data['regime_quality_weight']
    data['quality_enhanced_volume'] = data['fractal_volume_core'] * data['regime_quality_weight']
    
    # Multi-Timeframe Signal Synthesis
    # Short-term Signals
    data['short_term_momentum'] = data['regime_adapted_momentum'] * data['volume']
    data['volume_regime_congruence'] = np.sign(data['fractal_volume_core']) * np.sign(data['vol_regime_divergence'])
    data['short_term_core'] = data['short_term_momentum'] * data['volume_regime_congruence']
    
    # Medium-term Signals
    data['high_2d'] = data['high'].rolling(window=2, min_periods=1).max()
    data['low_2d'] = data['low'].rolling(window=2, min_periods=1).min()
    data['high_3d_prev'] = data['high'].rolling(window=3, min_periods=1).max().shift(3)
    data['low_3d_prev'] = data['low'].rolling(window=3, min_periods=1).min().shift(3)
    data['volatility_range_expansion'] = (data['high_2d'] - data['low_2d']) / (data['high_3d_prev'] - data['low_3d_prev'] + eps)
    
    data['vol_expansion_sign'] = (data['volatility_range_expansion'] > 1).astype(int)
    data['volatility_trend_consistency'] = data['vol_expansion_sign'].rolling(window=5, min_periods=1).sum()
    
    data['medium_term_momentum'] = data['regime_adapted_momentum'] * data['volume_persistence']
    data['medium_term_core'] = data['medium_term_momentum'] * data['volatility_trend_consistency']
    
    # Multi-Timeframe Alignment
    data['timeframe_alignment'] = np.sign(data['short_term_core']) * np.sign(data['medium_term_core'])
    data['aligned_multi_timeframe'] = data['short_term_core'] * data['medium_term_core']
    data['multi_timeframe_core'] = data['timeframe_alignment'] * data['aligned_multi_timeframe']
    
    # Final Alpha Construction
    # Core Signal Integration
    data['quality_enhanced_momentum_final'] = data['quality_enhanced_momentum'] * data['multi_timeframe_core']
    data['volume_exhaustion_component'] = data['quality_enhanced_volume'] * data['multi_timeframe_exhaustion_core']
    data['regime_quality_component'] = data['regime_quality_weight'] * data['volume_quality_score']
    
    # Signal Synthesis Framework
    data['momentum_volume_synthesis'] = data['quality_enhanced_momentum_final'] * data['volume_exhaustion_component']
    data['quality_regime_enhancement'] = data['momentum_volume_synthesis'] * data['regime_quality_component']
    data['multi_timeframe_adjustment'] = data['quality_regime_enhancement'] * data['multi_timeframe_core']
    
    # Final Alpha Generation
    data['alpha_core'] = data['multi_timeframe_adjustment'] * data['multi_timeframe_exhaustion_core']
    data['regime_adaptive_fractal_alpha'] = data['alpha_core'] * data['regime_quality_component']
    
    return data['regime_adaptive_fractal_alpha']
