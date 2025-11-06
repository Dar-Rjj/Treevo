import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Timeframe Price Fractals
    # Short-term Fractal (3-day)
    data['fractal_range_short'] = data['high'].rolling(window=3).max() - data['low'].rolling(window=3).min()
    data['fractal_efficiency_short'] = abs(data['close'] - data['close'].shift(2)) / data['fractal_range_short']
    data['fractal_momentum_short'] = (data['close'] - data['close'].shift(2)) / data['fractal_range_short']
    
    # Medium-term Fractal (8-day)
    data['fractal_range_medium'] = data['high'].rolling(window=8).max() - data['low'].rolling(window=8).min()
    data['fractal_efficiency_medium'] = abs(data['close'] - data['close'].shift(7)) / data['fractal_range_medium']
    data['fractal_momentum_medium'] = (data['close'] - data['close'].shift(7)) / data['fractal_range_medium']
    
    # Fractal Convergence
    data['fractal_alignment'] = np.sign(data['fractal_momentum_short']) * np.sign(data['fractal_momentum_medium'])
    data['efficiency_ratio'] = data['fractal_efficiency_short'] / data['fractal_efficiency_medium']
    data['fractal_strength'] = (data['fractal_momentum_short'] + data['fractal_momentum_medium']) * data['fractal_alignment']
    
    # Volume Fractal Patterns
    data['volume_avg_4d'] = data['volume'].shift(1).rolling(window=4).mean()
    data['volume_burst'] = data['volume'] / data['volume_avg_4d']
    
    # Volume Persistence (count of volume > previous volume for days t-4 to t)
    volume_persistence = []
    for i in range(len(data)):
        if i < 4:
            volume_persistence.append(0)
        else:
            count = 0
            for j in range(i-4, i+1):
                if j > 0 and data['volume'].iloc[j] > data['volume'].iloc[j-1]:
                    count += 1
            volume_persistence.append(count)
    data['volume_persistence'] = volume_persistence
    
    data['volume_momentum'] = (data['volume'] - data['volume'].shift(4)) / data['volume'].shift(4)
    data['volume_range_ratio'] = data['volume'] / (data['high'] - data['low'])
    data['fractal_volume_efficiency'] = data['fractal_efficiency_short'] * data['volume_burst']
    data['coupling_strength'] = data['fractal_strength'] * data['volume_persistence']
    
    # Fractal Regime Classification
    conditions = [
        (data['fractal_alignment'] > 0) & (data['efficiency_ratio'] > 1.2),
        (data['fractal_alignment'] < 0) & (data['efficiency_ratio'] < 0.8)
    ]
    choices = ['trending', 'mean_reverting']
    data['fractal_regime'] = np.select(conditions, choices, default='neutral')
    
    # Multi-Scale Volatility Waves
    # Short-term Volatility Wave (5-day)
    data['wave_amplitude_short'] = data['high'].rolling(window=5).max() - data['low'].rolling(window=5).min()
    data['wave_momentum_short'] = (data['close'] - data['close'].shift(4)) / data['wave_amplitude_short']
    data['wave_compression_short'] = data['wave_amplitude_short'] / (data['high'].rolling(window=5).max() - data['low'].rolling(window=5).min())
    
    # Medium-term Volatility Wave (15-day)
    data['wave_amplitude_medium'] = data['high'].rolling(window=15).max() - data['low'].rolling(window=15).min()
    data['wave_momentum_medium'] = (data['close'] - data['close'].shift(14)) / data['wave_amplitude_medium']
    data['wave_compression_medium'] = data['wave_amplitude_medium'] / (data['high'].rolling(window=15).max() - data['low'].rolling(window=15).min())
    
    # Wave Interference Patterns
    data['constructive_interference'] = np.sign(data['wave_momentum_short']) * np.sign(data['wave_momentum_medium'])
    data['wave_ratio'] = data['wave_amplitude_short'] / data['wave_amplitude_medium']
    data['interference_strength'] = (data['wave_momentum_short'] + data['wave_momentum_medium']) * data['constructive_interference']
    
    # Volume-Volatility Coupling
    data['volume_wave_ratio'] = data['volume'] / data['wave_amplitude_short']
    data['wave_volume_momentum'] = data['wave_momentum_short'] * data['volume_momentum']
    data['synchronization_strength'] = data['volume_wave_ratio'] * data['constructive_interference']
    
    # Pressure-Wave Dynamics
    data['buy_pressure_wave'] = (data['close'] - data['low']) / data['wave_amplitude_short']
    data['sell_pressure_wave'] = (data['high'] - data['close']) / data['wave_amplitude_short']
    data['net_pressure_wave'] = (data['buy_pressure_wave'] - data['sell_pressure_wave']) * data['volume_burst']
    
    # Wave Regime Detection
    wave_conditions = [
        (data['wave_ratio'] > 1.3) & (data['wave_compression_short'] > 0.8),
        (data['wave_ratio'] < 0.7) & (data['wave_compression_short'] < 0.5)
    ]
    wave_choices = ['high_amplitude', 'low_amplitude']
    data['wave_regime'] = np.select(wave_conditions, wave_choices, default='normal')
    
    # Microstructure Momentum Convergence
    data['structure_wave_convergence'] = data['fractal_strength'] * data['interference_strength']
    data['volume_structure_coupling'] = data['coupling_strength'] * data['synchronization_strength']
    data['pressure_structure_alignment'] = data['net_pressure_wave'] * data['fractal_alignment']
    
    # Momentum Persistence Patterns
    # Short-term consistency (count of sign consistency from t-3 to t)
    short_consistency = []
    for i in range(len(data)):
        if i < 3:
            short_consistency.append(0)
        else:
            signs = [np.sign(data['fractal_momentum_short'].iloc[j]) for j in range(i-3, i+1)]
            count = sum(1 for j in range(1, len(signs)) if signs[j] == signs[0])
            short_consistency.append(count)
    data['short_term_consistency'] = short_consistency
    
    # Medium-term consistency
    medium_consistency = []
    for i in range(len(data)):
        if i < 3:
            medium_consistency.append(0)
        else:
            signs = [np.sign(data['fractal_momentum_medium'].iloc[j]) for j in range(i-3, i+1)]
            count = sum(1 for j in range(1, len(signs)) if signs[j] == signs[0])
            medium_consistency.append(count)
    data['medium_term_consistency'] = medium_consistency
    
    # Wave consistency
    wave_consistency = []
    for i in range(len(data)):
        if i < 3:
            wave_consistency.append(0)
        else:
            signs = [np.sign(data['wave_momentum_short'].iloc[j]) for j in range(i-3, i+1)]
            count = sum(1 for j in range(1, len(signs)) if signs[j] == signs[0])
            wave_consistency.append(count)
    data['wave_consistency'] = wave_consistency
    
    data['volume_momentum_correlation'] = np.sign(data['volume_momentum']) * np.sign(data['fractal_momentum_short'])
    data['persistence_strength'] = (data['short_term_consistency'] + data['medium_term_consistency']) * data['volume_momentum_correlation']
    data['momentum_acceleration'] = data['fractal_momentum_short'] - data['fractal_momentum_medium']
    
    # Convergence Signal Construction
    data['base_convergence'] = data['structure_wave_convergence'] * data['volume_structure_coupling']
    data['persistence_enhancement'] = data['base_convergence'] * data['persistence_strength']
    data['acceleration_component'] = data['persistence_enhancement'] * data['momentum_acceleration']
    
    # Regime-Adaptive Signal Synthesis
    # Fractal Regime Adjustments
    regime_signals = []
    for i in range(len(data)):
        fractal_regime = data['fractal_regime'].iloc[i]
        acc_component = data['acceleration_component'].iloc[i]
        fractal_strength = data['fractal_strength'].iloc[i]
        vol_structure_coupling = data['volume_structure_coupling'].iloc[i]
        volume_persistence_val = data['volume_persistence'].iloc[i]
        interference_strength = data['interference_strength'].iloc[i]
        constructive_interference = data['constructive_interference'].iloc[i]
        
        if fractal_regime == 'trending':
            enhanced_momentum = acc_component * fractal_strength
            volume_confirmation = vol_structure_coupling * volume_persistence_val
            wave_alignment = interference_strength * constructive_interference
            regime_signal = (enhanced_momentum + volume_confirmation + wave_alignment) / 3
        elif fractal_regime == 'mean_reverting':
            momentum_reversal = -acc_component * fractal_strength
            volume_divergence = vol_structure_coupling / (volume_persistence_val + 1e-8)
            wave_counter_alignment = interference_strength / (constructive_interference + 1e-8)
            regime_signal = (momentum_reversal + volume_divergence + wave_counter_alignment) / 3
        else:  # neutral
            balanced_momentum = acc_component
            standard_volume = vol_structure_coupling
            neutral_wave = interference_strength
            regime_signal = (balanced_momentum + standard_volume + neutral_wave) / 3
        
        regime_signals.append(regime_signal)
    
    data['regime_signal'] = regime_signals
    
    # Wave Regime Multipliers
    final_signals = []
    for i in range(len(data)):
        regime_signal = data['regime_signal'].iloc[i]
        wave_regime = data['wave_regime'].iloc[i]
        wave_ratio = data['wave_ratio'].iloc[i]
        volume_confirmation = data['volume_structure_coupling'].iloc[i] * data['volume_persistence'].iloc[i]
        volume_burst_val = data['volume_burst'].iloc[i]
        pressure_structure_alignment = data['pressure_structure_alignment'].iloc[i]
        net_pressure_wave = data['net_pressure_wave'].iloc[i]
        
        if wave_regime == 'high_amplitude':
            volatility_boost = regime_signal * wave_ratio
            volume_amplification = volume_confirmation * volume_burst_val
            pressure_enhancement = pressure_structure_alignment * net_pressure_wave
            final_signal = (volatility_boost + volume_amplification + pressure_enhancement) / 3
        elif wave_regime == 'low_amplitude':
            volatility_dampening = regime_signal / (wave_ratio + 1e-8)
            volume_stabilization = volume_confirmation / (volume_burst_val + 1e-8)
            pressure_normalization = pressure_structure_alignment / (abs(net_pressure_wave) + 1e-8)
            final_signal = (volatility_dampening + volume_stabilization + pressure_normalization) / 3
        else:  # normal
            standard_volatility = regime_signal
            normal_volume = volume_confirmation
            base_pressure = pressure_structure_alignment
            final_signal = (standard_volatility + normal_volume + base_pressure) / 3
        
        final_signals.append(final_signal)
    
    # Return the final alpha factor
    return pd.Series(final_signals, index=data.index, name='alpha_factor')
