import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Fractal Volatility-Efficiency Core
    # Fractal Range Efficiency
    data['range'] = data['high'] - data['low']
    data['prev_range'] = data['range'].shift(1)
    data['fractal_range_ratio'] = data['range'] / data['prev_range']
    
    data['range_efficiency'] = data['volume'] / np.maximum(
        data['range'], 
        np.maximum(
            np.abs(data['high'] - data['close'].shift(1)),
            np.abs(data['low'] - data['close'].shift(1))
        )
    )
    data['fractal_efficiency_momentum'] = data['range_efficiency'] - data['range_efficiency'].shift(1)
    
    # Volume-Volatility Fractal Dynamics
    data['prev_volume'] = data['volume'].shift(1)
    data['volume_fractal_ratio'] = data['volume'] / data['prev_volume']
    
    data['volume_volatility_fractal'] = (data['volume'] / data['range']) - (data['prev_volume'] / data['prev_range'])
    data['volume_efficiency_flow'] = data['volume_fractal_ratio'] * data['range_efficiency']
    
    # Price Microstructure Fractals
    data['fractal_closing_pattern'] = (data['close'] - data['open']) / data['range']
    data['gap_efficiency'] = (data['close'] - data['open']) / np.abs(data['open'] - data['close'].shift(1))
    data['intraday_fractal_efficiency'] = (data['close'] - data['open']) / np.maximum(
        data['range'], 
        np.abs(data['open'] - data['close'].shift(1))
    )
    
    # Multi-Scale Regime Detection
    # Volatility Regime Classification
    data['volatility_breakout'] = (data['range'] > 1.5 * data['prev_range']).astype(int)
    data['volatility_breakout_count'] = data['volatility_breakout'].rolling(window=5, min_periods=1).sum()
    data['intraday_volatility_surge'] = data['range'] / data['prev_range']
    
    data['range_compression'] = data['range'] / data['range'].shift(4)
    data['contraction_count'] = (data['range'] < data['prev_range']).astype(int)
    data['contraction_persistence'] = data['contraction_count'].rolling(window=5, min_periods=1).sum()
    
    data['expansion_score'] = data['volatility_breakout_count'] / 4
    data['contraction_score'] = data['contraction_persistence'] / 4
    data['regime_signal'] = data['expansion_score'] - data['contraction_score']
    
    # Fractal Regime Patterns
    data['expanding_fractal_regime'] = ((data['fractal_range_ratio'] > 1.5) & (data['volume_fractal_ratio'] > 1.2)).astype(int)
    data['contracting_fractal_regime'] = ((data['fractal_range_ratio'] < 0.8) & (data['volume_fractal_ratio'] < 0.9)).astype(int)
    
    # Multi-Scale Regime Integration
    data['medium_term_regime'] = data['regime_signal'].rolling(window=5, min_periods=1).mean()
    
    data['fractal_regime_alignment'] = np.where(
        data['expanding_fractal_regime'] == 1,
        data['expansion_score'],
        np.where(
            data['contracting_fractal_regime'] == 1,
            data['contraction_score'],
            0
        )
    )
    
    # Fractal Divergence Patterns
    # Price-Volume Fractal Divergence
    data['price_change'] = data['close'] - data['close'].shift(1)
    data['volume_change'] = data['volume'] - data['prev_volume']
    
    data['negative_divergence'] = np.where(
        np.sign(data['price_change']) != np.sign(data['volume_change']),
        data['price_change'] / data['volume'],
        0
    )
    data['positive_divergence'] = np.where(
        np.sign(data['price_change']) == np.sign(data['volume_change']),
        data['price_change'] / data['volume'],
        0
    )
    
    data['divergence_intensity'] = np.abs(data['negative_divergence'] - data['positive_divergence'])
    
    data['high_volume_low_vol'] = np.where(
        (data['volume'] > data['prev_volume']) & (data['range'] < data['prev_range']),
        data['volume'] / data['range'],
        0
    )
    data['low_volume_high_vol'] = np.where(
        (data['volume'] < data['prev_volume']) & (data['range'] > data['prev_range']),
        data['range'] / data['volume'],
        0
    )
    data['dislocation_score'] = data['high_volume_low_vol'] - data['low_volume_high_vol']
    
    # Fractal Momentum Patterns
    data['fractal_momentum_persistence'] = (
        np.sign(data['close'] - data['open']) * 
        np.sign(data['close'].shift(1) - data['open'].shift(1)) * 
        np.sign(data['close'].shift(2) - data['open'].shift(2))
    )
    data['fractal_gap_momentum'] = (
        np.sign(data['open'] - data['close'].shift(1)) * 
        np.sign(data['open'].shift(1) - data['close'].shift(2)) * 
        np.sign(data['open'].shift(2) - data['close'].shift(3))
    )
    data['combined_fractal_momentum'] = data['fractal_momentum_persistence'] + data['fractal_gap_momentum']
    
    # Fractal Efficiency Divergence
    data['price_fractal_efficiency'] = np.abs(data['close'] - data['open']) / data['range']
    data['volume_fractal_efficiency'] = (data['volume'] / data['range']) * np.abs(data['close'] - data['open'])
    data['fractal_alignment_score'] = (
        np.sign(data['close'] - data['open']) * 
        np.sign(data['volume'] - data['prev_volume']) * 
        data['price_fractal_efficiency']
    )
    
    # Multi-Scale Signal Construction
    # Short-term Components
    data['immediate_regime_strength'] = data['regime_signal'] * data['price_change']
    data['volatility_momentum'] = (data['range'] / data['prev_range']) * data['regime_signal']
    data['efficiency_momentum'] = data['volume_efficiency_flow'] * data['fractal_efficiency_momentum']
    
    # Medium-term Components
    data['volume_correlation'] = (
        data['volume'].rolling(window=5, min_periods=1).apply(
            lambda x: np.sum(x * data.loc[x.index, 'regime_signal']) / np.sum(x)
        )
    )
    
    data['volatility_persistence'] = (data['range'] > data['prev_range']).astype(int)
    data['volatility_persistence_count'] = data['volatility_persistence'].rolling(window=5, min_periods=1).sum()
    data['vwap_ratio'] = (data['amount'] / data['volume']) / (data['amount'].shift(1) / data['prev_volume'])
    data['volatility_persistence_score'] = data['volatility_persistence_count'] * data['vwap_ratio']
    
    # Timeframe Integration
    data['short_term_engine'] = data['immediate_regime_strength'] * data['volatility_momentum'] * data['efficiency_momentum']
    data['medium_term_engine'] = data['medium_term_regime'] * data['volume_correlation'] * data['volatility_persistence_score']
    data['multi_scale_alignment'] = data['short_term_engine'] * data['medium_term_engine']
    
    # Adaptive Enhancement
    data['expansion_multiplier'] = 1 + data['expansion_score']
    data['contraction_multiplier'] = 1 + data['contraction_score']
    data['transition_multiplier'] = 1 + np.abs(data['regime_signal'])
    
    data['expansion_signal'] = data['divergence_intensity'] * data['expansion_multiplier']
    data['contraction_signal'] = data['dislocation_score'] * data['contraction_multiplier']
    data['transition_signal'] = data['divergence_intensity'] * data['transition_multiplier']
    
    data['expansion_core'] = data['expansion_signal'] * data['fractal_regime_alignment']
    data['contraction_core'] = data['contraction_signal'] * data['fractal_regime_alignment']
    data['transition_core'] = data['transition_signal'] * data['regime_signal']
    
    # Alpha Construction
    data['core_engine'] = (
        data['multi_scale_alignment'] * 
        data['volume_fractal_efficiency'] * 
        data['gap_efficiency'] * 
        data['fractal_alignment_score']
    )
    
    data['regime_weighted_enhancement'] = (
        data['core_engine'] * 
        (data['expansion_core'] + data['contraction_core'] + data['transition_core'])
    )
    
    data['final_alpha'] = (
        data['regime_weighted_enhancement'] * 
        data['intraday_fractal_efficiency'] * 
        data['volume_volatility_fractal']
    )
    
    return data['final_alpha']
