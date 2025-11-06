import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Quantum Volatility State Synthesis
    # Short-term Volatility
    data['short_term_vol'] = (data['high'] - data['low']) / (data['close'].shift(1) + 0.001)
    
    # Medium-term Volatility using rolling windows
    data['high_5d'] = data['high'].rolling(window=5).max()
    data['low_5d'] = data['low'].rolling(window=5).min()
    data['medium_term_vol'] = (data['high_5d'] - data['low_5d']) / (data['close'].shift(5) + 0.001)
    
    # Volatility State Uncertainty
    data['vol_uncertainty'] = abs(data['short_term_vol'] - data['medium_term_vol'])
    
    # Quantum Volume-Price Measurement
    # Price State Certainty
    high_low_range = data['high'] - data['low'] + 0.001
    data['price_certainty'] = abs(
        (data['close'] - data['low']) / high_low_range - 
        (data['high'] - data['close']) / high_low_range
    )
    
    # Volume State Momentum
    data['vol_momentum_1'] = data['volume'] / data['volume'].shift(1)
    data['vol_momentum_2'] = data['volume'].shift(1) / data['volume'].shift(2)
    data['volume_state_momentum'] = data['vol_momentum_1'] - data['vol_momentum_2']
    
    # Quantum Measurement Strength
    data['quantum_measurement'] = data['price_certainty'] * data['volume_state_momentum']
    
    # Volatility-Quantum Entanglement
    data['state_correlation'] = data['vol_uncertainty'] * data['quantum_measurement']
    data['entanglement_efficiency'] = 1 / (data['state_correlation'] + 0.001)
    
    # Volatility Ratio for Quantum Volatility Signal
    data['volatility_ratio'] = data['short_term_vol'] / (data['medium_term_vol'] + 0.001)
    data['quantum_vol_signal'] = data['entanglement_efficiency'] * data['volatility_ratio']
    
    # Harmonic Divergence Dynamics
    # Asymmetric Pressure Components
    prev_high_low_range = data['high'].shift(1) - data['low'].shift(1) + 0.001
    data['opening_pressure'] = (data['open'] - data['close'].shift(1)) / prev_high_low_range
    data['closing_pressure'] = (data['close'] - data['open']) / high_low_range
    
    # Pressure Divergence
    pressure_diff = data['opening_pressure'] - data['closing_pressure']
    data['pressure_divergence'] = data['opening_pressure'] * data['closing_pressure'] * np.sign(pressure_diff)
    
    # Volume-Weighted Momentum
    data['price_momentum'] = (data['close'] - data['close'].shift(1)) / prev_high_low_range
    data['volume_momentum'] = (data['volume'] - data['volume'].shift(1)) / (data['volume'].shift(1) + 0.001)
    
    # Momentum Divergence
    momentum_diff = data['price_momentum'] - data['volume_momentum']
    data['momentum_divergence'] = data['price_momentum'] * data['volume_momentum'] * np.sign(momentum_diff)
    
    # Harmonic Convergence Analysis
    data['pressure_momentum_alignment'] = np.sign(data['pressure_divergence']) * np.sign(data['momentum_divergence'])
    data['divergence_coherence'] = abs(data['pressure_divergence'] - data['momentum_divergence'])
    data['harmonic_divergence_strength'] = data['pressure_momentum_alignment'] * (1 - data['divergence_coherence'])
    
    # Fractal Regime Synchronization
    # Multi-Timeframe Fractal Patterns
    data['close_diff_5'] = abs(data['close'] - data['close'].shift(5))
    
    # Calculate sum of absolute price changes over 5 days
    data['price_change_sum'] = 0
    for i in range(5):
        data['price_change_sum'] += abs(data['close'].shift(i) - data['close'].shift(i+1))
    
    data['short_term_fractal_eff'] = data['close_diff_5'] / (data['price_change_sum'] + 0.001)
    data['medium_term_momentum_persist'] = (data['close'] - data['close'].shift(10)) / (data['close'].shift(10) + 0.001)
    data['fractal_synchronization'] = data['short_term_fractal_eff'] * data['medium_term_momentum_persist']
    
    # Volume Fractal Dynamics
    data['volume_fractal_density'] = (data['volume'] / data['volume'].shift(3)) - (data['volume'] / data['volume'].shift(5))
    data['volume_persistence'] = data['volume'] / data['volume'].shift(10)
    data['volume_fractal_sync'] = data['volume_fractal_density'] * data['volume_persistence']
    
    # Fractal Regime Classification
    data['high_sync'] = (data['fractal_synchronization'] > 0.1) & (data['volume_fractal_sync'] > 0)
    data['low_sync'] = (data['fractal_synchronization'] < 0.05) & (data['volume_fractal_sync'] < 0)
    data['transition_regime'] = abs(data['fractal_synchronization'] - data['volume_fractal_sync']) > 0.2
    
    # Adaptive Quantum Harmonic Fusion
    # Core Quantum Harmonic Signal
    data['quantum_base'] = data['quantum_vol_signal'] * data['harmonic_divergence_strength']
    data['fractal_enhancement'] = data['quantum_base'] * data['fractal_synchronization']
    data['volume_confirmation'] = data['fractal_enhancement'] * np.sign(data['volume_momentum'])
    
    # Regime-Adaptive Weighting
    data['regime_weight'] = data['volume_confirmation'].copy()
    data.loc[data['high_sync'], 'regime_weight'] = data['volume_confirmation'] * 1.3
    data.loc[data['low_sync'], 'regime_weight'] = data['volume_confirmation'] * 0.8
    data.loc[data['transition_regime'], 'regime_weight'] = data['volume_confirmation'] * data['pressure_divergence']
    
    # Volatility Adjustment
    data['volatility_adjusted_signal'] = data['regime_weight'].copy()
    high_vol_mask = data['volatility_ratio'] > 1
    data.loc[high_vol_mask, 'volatility_adjusted_signal'] = data['regime_weight'] * 1.2
    data.loc[~high_vol_mask, 'volatility_adjusted_signal'] = data['regime_weight'] * 0.9
    
    # Persistence Enhancement
    # Calculate signal persistence (consecutive same-sign days)
    data['signal_sign'] = np.sign(data['volatility_adjusted_signal'])
    data['persistence_count'] = 0
    for i in range(1, len(data)):
        if data['signal_sign'].iloc[i] == data['signal_sign'].iloc[i-1]:
            data['persistence_count'].iloc[i] = data['persistence_count'].iloc[i-1] + 1
        else:
            data['persistence_count'].iloc[i] = 1
    
    data['amplitude_modulation'] = data['volatility_adjusted_signal'] * abs(data['price_momentum'])
    data['quantum_harmonic_alpha'] = data['amplitude_modulation'] * data['persistence_count']
    
    # Final Alpha Factor
    data['volatility_adaptive_quantum_div'] = data['quantum_harmonic_alpha'] * data['entanglement_efficiency']
    data['final_alpha'] = data['volatility_adaptive_quantum_div'] * data['fractal_synchronization']
    
    return data['final_alpha']
