import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # Calculate basic price metrics
    data['TrueRange'] = np.maximum(data['high'] - data['low'], 
                                  np.maximum(abs(data['high'] - data['close'].shift(1)), 
                                            abs(data['low'] - data['close'].shift(1))))
    
    # Multi-Timeframe Momentum Decoherence
    # Momentum Wave Functions
    data['short_momentum'] = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    data['medium_momentum'] = (data['close'] - data['close'].shift(10)) / data['close'].shift(10)
    data['long_momentum'] = (data['close'] - data['close'].shift(20)) / data['close'].shift(20)
    data['momentum_coherence'] = 1 - abs(data['short_momentum'] - data['medium_momentum'])
    
    # Breakout-Induced Decoherence
    data['5d_high_breakout'] = data['close'] > data['high'].rolling(window=5, min_periods=5).apply(lambda x: x[:-1].max() if len(x) > 1 else np.nan)
    data['10d_high_breakout'] = data['close'] > data['high'].rolling(window=10, min_periods=10).apply(lambda x: x[:-1].max() if len(x) > 1 else np.nan)
    data['20d_high_breakout'] = data['close'] > data['high'].rolling(window=20, min_periods=20).apply(lambda x: x[:-1].max() if len(x) > 1 else np.nan)
    
    # Breakout persistence
    data['breakout_count'] = data['5d_high_breakout'].astype(int) + data['10d_high_breakout'].astype(int) + data['20d_high_breakout'].astype(int)
    data['breakout_persistence'] = data['breakout_count'].rolling(window=5, min_periods=1).sum()
    
    # Trade-induced collapse
    data['amount_5d_avg'] = data['amount'].rolling(window=5, min_periods=5).mean()
    data['trade_induced_collapse'] = data['amount'] / data['amount_5d_avg']
    
    # Wave function collapse
    data['wave_function_collapse'] = abs(data['close'] - data['open']) / data['TrueRange']
    
    # Thermodynamic Volume-Range Propagation
    # Range Thermodynamics
    data['range_efficiency'] = abs(data['close'] - data['open']) / (data['high'] - data['low'])
    data['range_5d_avg'] = (data['high'] - data['low']).rolling(window=5, min_periods=5).mean()
    data['range_expansion'] = (data['high'] - data['low']) / data['range_5d_avg'].shift(1)
    data['entropy_production'] = (data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1))
    
    # Volume as Acoustic Propagation
    data['volume_momentum'] = (data['volume'] - data['volume'].shift(5)) / data['volume'].shift(5)
    data['overnight_gap'] = abs(data['open'] - data['close'].shift(1))
    data['acoustic_impedance'] = data['overnight_gap'] / data['TrueRange']
    data['propagation_efficiency'] = data['range_efficiency'] * data['volume_momentum']
    
    # Morphogenetic Breakout Patterning
    # Trade Size Developmental Fields
    data['trade_size_ratio'] = data['amount'] / (data['volume'] * data['close'])
    data['trade_size_5d_avg'] = data['trade_size_ratio'].rolling(window=5, min_periods=5).mean()
    data['morphogen_gradient'] = data['trade_size_ratio'] / data['trade_size_5d_avg']
    
    # Breakout Pattern Formation
    data['breakout_strength'] = data['breakout_count'] * data['short_momentum']
    data['volatility_ratio'] = (data['high'] - data['low']).rolling(window=5).std() / (data['high'] - data['low']).rolling(window=20).std()
    data['pattern_stability'] = data['volatility_ratio'] / data['morphogen_gradient']
    
    # Gravitational-Divergence Lensing
    # Spacetime Curvature Effects
    data['gravitational_potential'] = (data['high'] - data['low']) / data['close']
    data['momentum_redshift'] = data['short_momentum'] - data['medium_momentum']
    
    # Volume-Confirmed Divergence
    data['volume_5d_avg'] = data['volume'].rolling(window=5, min_periods=5).mean()
    data['volume_range_alignment'] = data['range_expansion'] * data['volume'] / data['volume_5d_avg']
    data['divergence_patterns'] = data['volume_momentum'] * (1 - data['range_efficiency'])
    data['stress_energy_distribution'] = data['amount'] / (data['TrueRange'] * data['close'])
    
    # Integrated Quantum-Thermodynamic Alpha
    # Decoherence-Propagation Integration
    data['quantum_efficiency'] = data['momentum_coherence'] * data['range_efficiency']
    data['thermodynamic_collapse'] = data['trade_induced_collapse'] * data['entropy_production']
    data['wave_particle_duality'] = data['quantum_efficiency'] * data['propagation_efficiency']
    
    # Breakout-Pattern Confirmation
    data['morphogenetic_validation'] = data['breakout_strength'] * data['pattern_stability']
    data['lensing_effect'] = data['momentum_redshift'] * data['gravitational_potential']
    data['gravitational_amplification'] = data['lensing_effect'] * data['volume_range_alignment']
    data['integrated_signal'] = data['wave_particle_duality'] * data['morphogenetic_validation']
    
    # Multi-Timeframe Signal Coherence
    data['timeframe_alignment'] = 1 - abs(data['short_momentum'] - data['medium_momentum'])
    data['persistence_validation'] = data['breakout_persistence'] * (1 / data['volatility_ratio'])
    
    # Final alpha factor
    data['alpha_factor'] = (data['integrated_signal'] * 
                           data['timeframe_alignment'] * 
                           data['persistence_validation'])
    
    # Clean up and return
    alpha_series = data['alpha_factor'].replace([np.inf, -np.inf], np.nan).fillna(0)
    return alpha_series
