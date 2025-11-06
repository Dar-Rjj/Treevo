import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Scale Volatility-Coherence Structure
    # Volatility-Weighted Coherence Momentum
    data['raw_coherence_momentum'] = ((data['close'] - data['close'].shift(1)) / data['close'].shift(1)) * \
                                   ((data['close'] - data['open']) / (data['high'] - data['low']))
    
    data['volatility_scaling'] = (data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1))
    data['volatility_coherence_momentum'] = data['raw_coherence_momentum'] * data['volatility_scaling']
    
    # Volume-Coherence Interaction
    data['volume_change_ratio'] = data['volume'] / data['volume'].shift(1)
    data['volume_price_alignment'] = np.sign(data['close'] - data['close'].shift(1)) * \
                                   np.sign(data['volume'] - data['volume'].shift(1))
    data['volume_coherence_momentum'] = data['volume_change_ratio'] * data['volume_price_alignment'] * data['volatility_scaling']
    
    # Fractal Momentum Decay
    data['coherence_acceleration'] = (
        ((data['close'] - data['close'].shift(1)) / data['close'].shift(1)) * 
        ((data['close'] - data['open']) / (data['high'] - data['low'])) -
        ((data['close'].shift(2) - data['close'].shift(3)) / data['close'].shift(3)) * 
        ((data['close'].shift(2) - data['open'].shift(2)) / (data['high'].shift(2) - data['low'].shift(2)))
    )
    data['volatility_weighted_acceleration'] = data['coherence_acceleration'] * data['volatility_scaling']
    data['fractal_decay_factor'] = data['volatility_weighted_acceleration'] * (data['volume'] / data['volume'].shift(2))
    
    # Adaptive Fractal Regime Detection
    # Volatility-Coherence Regime Classification
    data['high_volatility_coherent'] = (
        (data['high'] - data['low']) > (data['high'].shift(1) - data['low'].shift(1)) * 1.5
    ) & (
        (data['close'] - data['open']) / (data['high'] - data['low']) > 0.7
    )
    
    data['low_volatility_coherent'] = (
        (data['high'] - data['low']) < (data['high'].shift(1) - data['low'].shift(1)) * 0.7
    ) & (
        (data['close'] - data['open']) / (data['high'] - data['low']) > 0.7
    )
    
    data['high_volatility_incoherent'] = (
        (data['high'] - data['low']) > (data['high'].shift(1) - data['low'].shift(1)) * 1.5
    ) & (
        (data['close'] - data['open']) / (data['high'] - data['low']) < 0.3
    )
    
    data['low_volatility_incoherent'] = (
        (data['high'] - data['low']) < (data['high'].shift(1) - data['low'].shift(1)) * 0.7
    ) & (
        (data['close'] - data['open']) / (data['high'] - data['low']) < 0.3
    )
    
    # Volume-Fractal Regime Classification
    vol_ratio = data['volume'] / data['volume'].shift(1)
    vol_vol_ratio = np.log(data['volume'] / data['volume'].shift(1)) / \
                   np.log((data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1)) + 1e-8)
    
    data['high_volume_fractal'] = (vol_ratio > 1.8) & (vol_vol_ratio > 1.2)
    data['low_volume_fractal'] = (vol_ratio < 0.6) & (vol_vol_ratio < 0.8)
    data['normal_volume_fractal'] = ~data['high_volume_fractal'] & ~data['low_volume_fractal']
    
    # Price-Coherence Momentum Regime
    data['strong_coherent_up'] = (
        (data['close'] > data['close'].shift(1)) & 
        ((data['close'] - data['open']) / (data['high'] - data['low']) > 0.7) & 
        (data['volume_price_alignment'] > 0)
    )
    
    data['strong_coherent_down'] = (
        (data['close'] < data['close'].shift(1)) & 
        ((data['close'] - data['open']) / (data['high'] - data['low']) > 0.7) & 
        (data['volume_price_alignment'] < 0)
    )
    
    data['mixed_coherence'] = ~data['strong_coherent_up'] & ~data['strong_coherent_down']
    
    # Fractal Regime-Adaptive Factor Construction
    # High Volatility Coherent Factor
    data['volatility_coherence_synergy'] = data['volatility_coherence_momentum'] * data['volume_change_ratio']
    
    # Coherence Confirmation (3-day rolling)
    coherence_ratio = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['coherence_confirmation'] = (
        (coherence_ratio > 0.7).rolling(window=3, min_periods=1).sum() / 3
    )
    
    data['high_volatility_coherent_factor'] = (
        data['volatility_coherence_synergy'] * 
        data['coherence_confirmation'] * 
        data['high_volatility_coherent'].astype(int) * 
        data['high_volume_fractal'].astype(int)
    )
    
    # Low Volatility Coherent Factor
    data['price_efficiency'] = abs(data['close'] - data['open']) / (data['high'] - data['low'])
    data['volume_consistency'] = data['volume'] / (
        data['volume'].shift(1) + data['volume'].shift(2) + data['volume'].shift(3)
    )
    
    data['low_volatility_coherent_factor'] = (
        data['price_efficiency'] * 
        data['volume_consistency'] * 
        data['low_volatility_coherent'].astype(int)
    )
    
    # Mixed Fractal Regime Factor
    data['cross_fractal_momentum'] = data['volatility_coherence_momentum'] * data['volume_coherence_momentum']
    
    # Fractal Stability Score (simplified)
    regime_cols = ['high_volatility_coherent', 'low_volatility_coherent', 
                   'high_volatility_incoherent', 'low_volatility_incoherent']
    data['current_regime'] = data[regime_cols].idxmax(axis=1)
    data['fractal_stability_score'] = (
        data['current_regime'] == data['current_regime'].shift(1)
    ).rolling(window=3, min_periods=1).mean()
    
    data['mixed_fractal_factor'] = (
        data['cross_fractal_momentum'] * 
        data['fractal_stability_score'] * 
        data['mixed_coherence'].astype(int)
    )
    
    # Dynamic Coherence Weighting System
    # Volatility-Coherence Weights
    data['high_volatility_weight'] = data['volatility_scaling']
    data['low_volatility_weight'] = 1 / (data['volatility_scaling'] + 1e-8)
    data['normal_volatility_weight'] = 1 - abs(data['volatility_scaling'] - 1)
    
    # Volume-Fractal Weights
    data['high_volume_weight'] = data['volume_change_ratio']
    data['low_volume_weight'] = 1 / (data['volume_change_ratio'] + 1e-8)
    data['normal_volume_weight'] = 1 - abs(data['volume_change_ratio'] - 1)
    
    # Coherence Momentum Weights
    close_coherence_ratio = (data['close'] - data['open']) / (data['high'] - data['low'])
    
    def count_coherent_up(window):
        return ((close_coherence_ratio.loc[window.index] > 0.7) & 
                (window > window.shift(1))).sum()
    
    def count_coherent_down(window):
        return ((close_coherence_ratio.loc[window.index] > 0.7) & 
                (window < window.shift(1))).sum()
    
    data['coherent_up_weight'] = data['close'].rolling(window=5).apply(count_coherent_up, raw=False) / 5
    data['coherent_down_weight'] = data['close'].rolling(window=5).apply(count_coherent_down, raw=False) / 5
    data['neutral_coherence_weight'] = 1 - abs(data['coherent_up_weight'] - data['coherent_down_weight'])
    
    # Multi-Scale Coherence Integration
    # Weighted Fractal Components
    data['high_regime_component'] = (
        data['high_volatility_coherent_factor'] * 
        data['high_volatility_weight'] * 
        data['high_volume_weight']
    )
    
    data['low_regime_component'] = (
        data['low_volatility_coherent_factor'] * 
        data['low_volatility_weight'] * 
        data['low_volume_weight']
    )
    
    data['mixed_regime_component'] = (
        data['mixed_fractal_factor'] * 
        data['normal_volatility_weight'] * 
        data['normal_volume_weight']
    )
    
    # Coherence-Weighted Integration
    data['coherent_up_integration'] = (
        (data['high_regime_component'] + data['mixed_regime_component']) * 
        data['coherent_up_weight']
    )
    
    data['coherent_down_integration'] = (
        (data['low_regime_component'] + data['mixed_regime_component']) * 
        data['coherent_down_weight']
    )
    
    data['neutral_coherence_integration'] = (
        (data['high_regime_component'] + data['low_regime_component'] + data['mixed_regime_component']) * 
        data['neutral_coherence_weight']
    )
    
    # Fractal Factor Synthesis
    data['volatility_coherence_adaptive'] = (
        data['coherent_up_integration'] + 
        data['coherent_down_integration'] + 
        data['neutral_coherence_integration']
    )
    
    data['volume_fractal_confirmation'] = data['volatility_coherence_adaptive'] * data['volume_change_ratio']
    data['efficiency_coherence_adjustment'] = data['volume_fractal_confirmation'] * data['price_efficiency']
    
    # Breakout & Coherence Enhancement
    # Volatility-Coherence Breakout Signals
    data['range_expansion'] = (data['high'] - data['low']) / (
        (data['high'].shift(1) - data['low'].shift(1) + 
         data['high'].shift(2) - data['low'].shift(2) + 
         data['high'].shift(3) - data['low'].shift(3) + 
         data['high'].shift(4) - data['low'].shift(4)) / 4
    )
    
    data['coherence_breakout_strength'] = (
        (data['close'] - data['open']) / (data['high'] - data['low']) * 
        data['range_expansion']
    )
    
    data['volume_fractal_confirmation_breakout'] = data['volume_change_ratio'] * data['range_expansion']
    
    # Fractal Reversal Patterns
    data['coherence_exhaustion'] = (
        abs(data['volume_price_alignment']) > 
        2 * abs((data['close'] - data['open']) / (data['high'] - data['low']))
    ).astype(int)
    
    data['volume_coherence_divergence'] = (
        data['volume_price_alignment'] * 
        (data['close'] - data['open']) / (data['high'] - data['low'])
    )
    
    data['efficiency_reversal'] = data['price_efficiency'] * data['coherence_exhaustion']
    
    # Final Alpha Generation
    data['core_fractal_alpha'] = data['efficiency_coherence_adjustment'] * data['fractal_decay_factor']
    data['volatility_coherence_resilience'] = data['core_fractal_alpha'] / (data['volatility_scaling'] + 1e-8)
    data['breakout_integration'] = data['volatility_coherence_resilience'] * data['coherence_breakout_strength']
    data['final_adaptive_fractal_alpha'] = data['breakout_integration'] * data['coherence_confirmation']
    
    # Return the final alpha factor
    return data['final_adaptive_fractal_alpha']
