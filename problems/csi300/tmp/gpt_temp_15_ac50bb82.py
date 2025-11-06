import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Multi-Scale Entropic Resonance Framework
    data['short_term_entropic_resonance'] = (np.abs(data['close'] - data['open']) * data['volume'] / 
                                            (data['high'] - data['low']).rolling(window=3).apply(lambda x: (x * data.loc[x.index, 'volume']).sum()) * 
                                            np.log(data['volume'] / data['volume'].shift(1) + 1))
    
    data['medium_term_entropic_resonance'] = (np.abs(data['close'] - data['open']) * data['volume'] / 
                                             (data['high'] - data['low']).rolling(window=10).apply(lambda x: (x * data.loc[x.index, 'volume']).sum()) * 
                                             np.log(data['volume'].shift(5) / data['volume'].shift(6) + 1))
    
    data['entropic_resonance_convergence'] = (data['medium_term_entropic_resonance'] - data['short_term_entropic_resonance']) * np.sign(data['close'] - data['open'])
    
    # Quantum Volatility Microstructure
    data['momentum_wave'] = ((data['close'] - data['close'].shift(1)) * 
                            np.log(data['volume'] + 1) / np.log(data['volume'].shift(1) + 1))
    
    data['volatility_particle'] = ((data['high'] - data['low']) * (data['open'] - data['close'].shift(1)) / 
                                  (np.abs(data['close'] - data['open']) + 0.001))
    
    data['quantum_entropic_state'] = data['momentum_wave'] * data['volatility_particle']
    
    data['volume_certainty'] = data['volume'] / (data['high'] - data['low'] + 0.001)
    data['volatility_uncertainty'] = np.abs(data['close'] - data['open']) / (data['high'] - data['low'] + 0.001)
    data['entropic_uncertainty_product'] = data['volume_certainty'] * data['volatility_uncertainty']
    
    data['wave_particle_entropic_coupling'] = data['quantum_entropic_state'] * data['entropic_uncertainty_product']
    data['volume_volatility_entropic_correlation'] = (data['volume'] * data['volatility_particle'] / 
                                                     (data['high'] - data['low'] + 0.001))
    data['quantum_entropic_alpha'] = data['wave_particle_entropic_coupling'] * data['volume_volatility_entropic_correlation']
    
    # Spectral Entropic Momentum Architecture
    data['fast_entropic_momentum'] = (data['close'] - data['close'].shift(1)) * (data['volume'] / data['volume'].shift(1))
    data['medium_entropic_momentum'] = (data['close'] - data['close'].shift(3)) * (data['volume'] / data['volume'].shift(3))
    data['slow_entropic_momentum'] = (data['close'] - data['close'].shift(5)) * (data['volume'] / data['volume'].shift(5))
    
    data['fast_medium_entropic_phase'] = data['fast_entropic_momentum'] - data['medium_entropic_momentum']
    data['medium_slow_entropic_phase'] = data['medium_entropic_momentum'] - data['slow_entropic_momentum']
    data['entropic_phase_coherence'] = data['fast_medium_entropic_phase'] * data['medium_slow_entropic_phase']
    
    data['fast_medium_entropic_divergence'] = data['fast_entropic_momentum'] / (data['medium_entropic_momentum'] + 0.001)
    data['medium_slow_entropic_convergence'] = data['medium_entropic_momentum'] / (data['slow_entropic_momentum'] + 0.001)
    data['spectral_entropic_alpha'] = (data['entropic_phase_coherence'] * data['fast_medium_entropic_divergence'] * 
                                      data['medium_slow_entropic_convergence'])
    
    # Entropic Pressure-Resonance Framework
    data['opening_entropic_pressure'] = ((data['open'] - (data['high'].shift(1) + data['low'].shift(1)) / 2) / 
                                        ((data['high'].shift(1) - data['low'].shift(1)) / 2 + 0.001) * 
                                        (data['volume'] / data['volume'].shift(1) - 1))
    
    data['closing_entropic_pressure'] = ((data['close'] - (data['high'] + data['low']) / 2) / 
                                        ((data['high'] - data['low']) / 2 + 0.001) * 
                                        (data['volume'] / data['volume'].shift(1) - 1))
    
    data['total_entropic_pressure'] = data['opening_entropic_pressure'] + data['closing_entropic_pressure']
    
    data['entropic_resonance_volatility'] = ((data['high'] - data['low']) / (data['open'] + data['close'] + 0.001) * 
                                            data['volume'] / data['volume'].rolling(window=3).sum())
    
    data['entropic_pressure_resonance_state'] = data['total_entropic_pressure'] * data['entropic_resonance_volatility']
    
    # Quantum-Spectral Entropic Integration
    data['quantum_entropic_fusion'] = data['quantum_entropic_alpha'] * data['spectral_entropic_alpha']
    data['spectral_entropic_coupling'] = data['entropic_phase_coherence'] * data['entropic_resonance_volatility']
    data['volume_entropic_enhancement'] = (data['volume'] / data['volume'].rolling(window=3).sum() * 
                                          np.abs(data['close'] - data['open']) / (data['high'] - data['low'] + 0.001))
    
    data['base_quantum_spectral_entropic_alpha'] = (data['quantum_entropic_fusion'] * data['spectral_entropic_coupling'] * 
                                                   data['volume_entropic_enhancement'])
    
    # Quantum-Spectral Entropic State Assessment
    data['entropic_microstructure_strength'] = np.abs(data['quantum_entropic_state']) + np.abs(data['entropic_uncertainty_product'])
    data['spectral_entropic_strength'] = np.abs(data['fast_entropic_momentum']) + np.abs(data['entropic_phase_coherence'])
    data['entropic_resonance_strength'] = np.abs(data['entropic_resonance_volatility']) + np.abs(data['total_entropic_pressure'])
    data['volume_entropic_strength'] = (np.abs(data['volume'] / data['volume'].rolling(window=3).sum()) + 
                                       np.abs(data['volume_certainty']))
    
    total_strength = (data['entropic_microstructure_strength'] + data['spectral_entropic_strength'] + 
                     data['entropic_resonance_strength'] + data['volume_entropic_strength'])
    
    data['entropic_microstructure_weight'] = data['entropic_microstructure_strength'] / (total_strength + 0.001)
    data['spectral_entropic_weight'] = data['spectral_entropic_strength'] / (total_strength + 0.001)
    data['entropic_resonance_weight'] = data['entropic_resonance_strength'] / (total_strength + 0.001)
    data['volume_entropic_weight'] = data['volume_entropic_strength'] / (total_strength + 0.001)
    
    data['weighted_quantum_spectral_entropic_alpha'] = (
        data['quantum_entropic_alpha'] * data['entropic_microstructure_weight'] +
        data['spectral_entropic_alpha'] * data['spectral_entropic_weight'] +
        data['entropic_pressure_resonance_state'] * data['entropic_resonance_weight'] +
        (data['volume'] / data['volume'].rolling(window=3).sum()) * data['volume_entropic_weight']
    )
    
    # Dynamic Entropic Enhancement
    data['entropic_compression'] = (data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1) + 0.001)
    data['entropic_expansion'] = (data['high'].rolling(window=3).max() - data['low'].rolling(window=3).min()) / (data['high'].shift(1) - data['low'].shift(1) + 0.001)
    
    data['quantum_entropic_breakout_multiplier'] = 1 + ((data['entropic_compression'] < 0.6) * data['entropic_expansion'])
    
    data['volume_entropic_state'] = (data['weighted_quantum_spectral_entropic_alpha'] * 
                                    (data['volume'] / data['volume'].rolling(window=7).sum()))
    
    data['amount_entropic'] = data['volume_entropic_state'] * (data['amount'] / (data['volume'] + 0.001))
    
    data['quantum_entropic_resonance_alpha'] = (data['amount_entropic'] * data['quantum_entropic_breakout_multiplier'] * 
                                               data['entropic_resonance_convergence'])
    
    # Entropic Regime-Specific Enhancement
    high_entropic_condition = ((data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1) + 0.001) > 1.5) & \
                             ((data['high'] - data['low']) * (data['volume'] / data['volume'].shift(1)) > 1.2)
    
    low_entropic_condition = ((data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1) + 0.001) < 0.7) & \
                            ((data['high'] - data['low']) * (data['volume'] / data['volume'].shift(1)) < 0.8)
    
    expanding_entropic_condition = ((data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1) + 0.001) > 
                                   (data['high'] - data['low']).rolling(window=5).sum() / 
                                   (data['high'].shift(5) - data['low'].shift(5)).rolling(window=5).sum()) & \
                                  ((data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1) + 0.001) > 1.0)
    
    contracting_entropic_condition = ((data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1) + 0.001) < 
                                     (data['high'] - data['low']).rolling(window=5).sum() / 
                                     (data['high'].shift(5) - data['low'].shift(5)).rolling(window=5).sum()) & \
                                    ((data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1) + 0.001) < 1.0)
    
    # Apply regime enhancements
    regime_enhanced_alpha = data['quantum_entropic_resonance_alpha'].copy()
    
    regime_enhanced_alpha[high_entropic_condition] *= (
        (data['fast_entropic_momentum'] - data['medium_entropic_momentum']) * 
        (data['volume'] / data['volume'].shift(2))
    )[high_entropic_condition]
    
    regime_enhanced_alpha[low_entropic_condition] *= (
        data['total_entropic_pressure'] * 
        (data['volume'] / data['volume'].shift(1) - data['volume'].shift(1) / data['volume'].shift(2))
    )[low_entropic_condition]
    
    regime_enhanced_alpha[expanding_entropic_condition] *= (
        data['opening_entropic_pressure'] * 
        np.abs(data['close'] - data['open']) / (data['high'] - data['low'] + 0.001)
    )[expanding_entropic_condition]
    
    regime_enhanced_alpha[contracting_entropic_condition] *= (
        (data['volume'] / data['volume'].shift(1) - data['volume'].shift(1) / data['volume'].shift(2)) * 
        (data['high'] - data['low']) / (data['high'].shift(2) - data['low'].shift(2) + 0.001)
    )[contracting_entropic_condition]
    
    # Final Quantum Entropic Resonance Alpha
    final_alpha = regime_enhanced_alpha * (1 + np.abs(
        (data['fast_entropic_momentum'] - data['medium_entropic_momentum']) * 
        (data['medium_entropic_momentum'] - data['slow_entropic_momentum'])
    ))
    
    return final_alpha
