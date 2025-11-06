import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Scale Entropy-Pressure Synthesis
    # Price Entropy
    price_range = data['high'] - data['low']
    daily_move = np.abs(data['close'] - data['open']) + 1e-8
    price_entropy = (price_range / daily_move) * np.log(price_range / daily_move + 1e-8)
    
    # Volume Entropy
    volume_ratio = data['volume'] / data['volume'].shift(1) + 1e-8
    volume_diff_ratio = (data['volume'] - data['volume'].shift(1)) / (data['volume'] + data['volume'].shift(1) + 1e-8)
    volume_entropy = np.log(volume_ratio) * volume_diff_ratio
    
    # Multi-Scale Pressure Fields
    # Micro Pressure Field
    micro_pressure = ((data['close'] - data['low']) * (data['high'] - data['open'])) / ((data['high'] - data['low'])**2 + 1e-8)
    
    # Meso Pressure Field
    close_diff_3 = data['close'] - data['close'].shift(3)
    
    def rolling_volatility_product(window):
        if len(window) < 4:
            return np.nan
        return np.sum((window['high'] - window['low']) * window['volume'])
    
    vol_volatility_sum = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 3:
            window_data = data.iloc[i-3:i+1]
            vol_volatility_sum.iloc[i] = rolling_volatility_product(window_data)
        else:
            vol_volatility_sum.iloc[i] = np.nan
    
    meso_pressure = close_diff_3 * data['volume'] / (vol_volatility_sum + 1e-8)
    
    # Pressure Field Gradient
    pressure_gradient = micro_pressure - meso_pressure
    
    # Entropy-Pressure Core
    entropy_pressure_core = (price_entropy + volume_entropy) * pressure_gradient
    
    # Quantum Volume Flow Dynamics
    # Volume Memory Echo
    volume_memory_echo = data['volume'] / (data['volume'].shift(1) + data['volume'].shift(2) + 1e-8)
    
    # Volume Flow Momentum
    volume_flow_momentum = (data['volume'] / data['volume'].shift(1)) - (data['volume'].shift(1) / data['volume'].shift(2))
    
    # Amount Flow Efficiency
    amount_flow_efficiency = (data['amount'] / data['volume']) / (data['amount'].shift(1) / data['volume'].shift(1) + 1e-8)
    
    # Quantum Flow Core
    quantum_flow_core = volume_memory_echo * volume_flow_momentum * amount_flow_efficiency
    
    # Fractal Temporal Momentum
    # Multi-Scale Oscillation Components
    # High-Frequency Momentum
    high_freq_momentum = (data['close'] - data['close'].shift(1)) / (data['high'] - data['low'] + 1e-8)
    
    # Medium-Frequency Momentum
    def rolling_max_high(window):
        return window['high'].max()
    
    def rolling_min_low(window):
        return window['low'].min()
    
    medium_freq_range = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 2:
            window_data = data.iloc[i-2:i+1]
            max_high = rolling_max_high(window_data)
            min_low = rolling_min_low(window_data)
            medium_freq_range.iloc[i] = max_high - min_low
        else:
            medium_freq_range.iloc[i] = np.nan
    
    medium_freq_momentum = (data['close'] - data['close'].shift(3)) / (medium_freq_range + 1e-8)
    
    # Low-Frequency Momentum
    low_freq_range = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 4:
            window_data = data.iloc[i-4:i+1]
            max_high = rolling_max_high(window_data)
            min_low = rolling_min_low(window_data)
            low_freq_range.iloc[i] = max_high - min_low
        else:
            low_freq_range.iloc[i] = np.nan
    
    low_freq_momentum = (data['close'] - data['close'].shift(5)) / (low_freq_range + 1e-8)
    
    # Fractal Momentum Convergence
    phase_alignment = np.sign(high_freq_momentum) + np.sign(medium_freq_momentum) + np.sign(low_freq_momentum)
    momentum_resonance = phase_alignment * (high_freq_momentum + medium_freq_momentum + low_freq_momentum)
    
    # Quantum Regime Detection
    # Volatility Phase
    volatility_phase_range = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 4:
            window_data = data.iloc[i-4:i+1]
            max_high = rolling_max_high(window_data)
            min_low = rolling_min_low(window_data)
            volatility_phase_range.iloc[i] = max_high - min_low
        else:
            volatility_phase_range.iloc[i] = np.nan
    
    volatility_phase = (data['high'] - data['low']) / (volatility_phase_range + 1e-8)
    
    # Volume Phase
    volume_phase = data['volume'] / data['volume'].shift(4)
    
    # Quantum State Classification
    high_volatility_regime = (data['high'] - data['low']) > (2 * (data['high'].shift(1) - data['low'].shift(1)))
    low_volatility_regime = (data['high'] - data['low']) < (0.5 * (data['high'].shift(1) - data['low'].shift(1)))
    volume_breakout_regime = (data['volume'] > (1.8 * data['volume'].shift(1))) & (data['volume'] / data['volume'].shift(2) > 1.5)
    
    # Quantum Regime Multiplier
    base_multiplier = 1.0
    
    high_vol_enhancement = np.where(
        high_volatility_regime,
        1 + ((data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1) + 1e-8) - 1) * (data['volume'] / data['volume'].shift(1)),
        0
    )
    
    low_vol_dampening = np.where(
        low_volatility_regime,
        1 - (1 - (data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1) + 1e-8)) * (data['volume'] / data['volume'].shift(1)),
        0
    )
    
    volume_breakout_amplifier = np.where(
        volume_breakout_regime,
        1 + np.abs(data['close'] - data['close'].shift(1)) / (data['high'] - data['low'] + 1e-8) * (data['volume'] / data['volume'].shift(1)),
        0
    )
    
    regime_multiplier = base_multiplier + high_vol_enhancement + low_vol_dampening + volume_breakout_amplifier
    
    # Core Signal Construction
    quantum_fractal_base = entropy_pressure_core * quantum_flow_core
    momentum_enhanced_quantum = quantum_fractal_base * momentum_resonance
    regime_adapted_quantum = momentum_enhanced_quantum * regime_multiplier
    
    # Temporal-Fractal Confirmation
    price_breakout_confirmation = (data['close'] > data['high'].shift(1)) & ((data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8) > 0.6)
    phase_transition_confirmation = (volatility_phase > 0.5) | (volume_phase > 1.5)
    
    temporal_confirmation_multiplier = 1 + (price_breakout_confirmation.astype(int) + phase_transition_confirmation.astype(int)) * np.abs(data['close'] - data['close'].shift(1)) / (data['high'] - data['low'] + 1e-8)
    
    # Final Alpha Synthesis
    confirmed_quantum_alpha = regime_adapted_quantum * temporal_confirmation_multiplier
    
    typical_price = (data['open'] + data['high'] + data['low']) / 3
    price_deviation_ratio = np.abs(data['close'] - typical_price) / (data['high'] - data['low'] + 1e-8)
    
    quantum_fractal_alpha = confirmed_quantum_alpha * (1 - price_deviation_ratio)
    
    return quantum_fractal_alpha
