import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Multi-Scale Gap Absorption Dynamics
    for period in [3, 5, 10]:
        data[f'gap_size_{period}'] = (data['open'] - data['close'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1)).rolling(period).mean()
    
    data['quantum_absorption'] = (data['close'] - data['open']) / (abs(data['open'] - data['close'].shift(1)) + 1e-8)
    data['volume_participation'] = data['volume'] / (data['volume'].shift(1) + data['volume'].shift(2) + data['volume'].shift(3) + 1e-8)
    
    # Fractal Volatility-Efficiency Entanglement
    def calculate_atr(data, window):
        tr = np.maximum(data['high'] - data['low'], 
                       np.maximum(abs(data['high'] - data['close'].shift(1)), 
                                 abs(data['low'] - data['close'].shift(1))))
        return tr.rolling(window).mean()
    
    data['atr_5'] = calculate_atr(data, 5)
    data['atr_10'] = calculate_atr(data, 10)
    data['volatility_momentum_5'] = data['atr_5'] / data['atr_5'].shift(3) - 1
    data['volatility_momentum_10'] = data['atr_10'] / data['atr_10'].shift(3) - 1
    
    true_range = np.maximum(data['high'] - data['low'], 
                           np.maximum(abs(data['high'] - data['close'].shift(1)), 
                                     abs(data['low'] - data['close'].shift(1))))
    data['quantum_range_efficiency'] = abs(data['close'] - data['open']) / (true_range + 1e-8)
    data['volatility_efficiency_divergence'] = data['volatility_momentum_5'] * data['quantum_absorption']
    
    # Quantum Volume Acceleration Regimes
    data['volume_sign'] = np.sign(data['volume'] - data['volume'].shift(1))
    data['volume_persistence'] = data['volume_sign'].rolling(5).apply(lambda x: len(x[x == x.iloc[-1]]) if len(x) == 5 else np.nan)
    
    data['volume_momentum_5'] = data['volume'] / data['volume'].shift(5) - 1
    data['volume_momentum_10'] = data['volume'] / data['volume'].shift(10) - 1
    data['quantum_volume_acceleration'] = data['volume_momentum_5'] - data['volume_momentum_5'].shift(1)
    
    # Multi-Scale Pressure Accumulation
    for period in [3, 5]:
        data[f'cumulative_pressure_{period}'] = (data['volume'] * (data['close'] - data['open'])).rolling(period).sum()
    
    # Quantum-Fractal Regime Identification
    data['fractal_volatility_regime'] = data['close'].rolling(20).std() / data['close'].rolling(5).std()
    data['quantum_amplitude_regime'] = (data['high'] - data['low']) / (data['close'] + 1e-8)
    
    # Entangled Efficiency-Pressure Signals
    data['quantum_pressure_intensity'] = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'] + 1e-8)
    
    efficiency_3d = (abs(data['close'] - data['open']) / true_range).rolling(3).mean()
    efficiency_5d = (abs(data['close'] - data['open']) / true_range).rolling(5).mean()
    data['fractal_efficiency_momentum'] = efficiency_3d / (efficiency_5d + 1e-8)
    
    data['entangled_efficiency_pressure'] = data['quantum_pressure_intensity'] * data['fractal_efficiency_momentum']
    
    # Multi-Scale Breakout Confirmation
    data['fractal_breakout_magnitude'] = abs(data['close'] - data['high'].rolling(20).max()) / data['atr_10']
    data['quantum_pressure_acceleration'] = data['cumulative_pressure_5'] - data['cumulative_pressure_5'].shift(1)
    data['breakout_signal'] = data['fractal_breakout_magnitude'] * data['quantum_pressure_acceleration']
    
    # Volume-Efficiency Alignment Layer
    data['quantum_volume_clustering'] = data['volume'] / data['volume'].rolling(5).mean()
    data['fractal_efficiency_slope'] = efficiency_5d / data['atr_10']
    data['volume_efficiency_alignment'] = data['quantum_volume_clustering'] * data['fractal_efficiency_slope']
    
    # Composite Quantum-Fractal Alpha
    # Core Entangled Signal
    core_signal = (data['entangled_efficiency_pressure'] * 
                   data['quantum_pressure_acceleration'] * 
                   data['volatility_efficiency_divergence'] * 
                   data['quantum_volume_acceleration'])
    
    # Volume Confirmation Filter
    volume_filter = (data['volume_persistence'] * 
                     data['volume_participation'] * 
                     data['quantum_volume_clustering'])
    
    # Breakout Context Enhancement
    breakout_enhancement = (data['breakout_signal'] * 
                           data['volume_efficiency_alignment'] * 
                           data['cumulative_pressure_5'])
    
    # Final Signal Generation
    alpha_signal = (core_signal * volume_filter * breakout_enhancement)
    
    # Quantum Decoherence Filter
    volatility_filter = data['fractal_volatility_regime'].between(0.5, 2.0)
    efficiency_filter = data['fractal_efficiency_momentum'] > 0.3
    volume_strength_filter = data['quantum_volume_clustering'] > 0.7
    
    final_alpha = alpha_signal * volatility_filter * efficiency_filter * volume_strength_filter
    
    return final_alpha
