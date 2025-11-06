import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Entropic Microstructure Fragmentation
    # 5-day high-low entropy divergence
    high_low_range = df['high'] - df['low']
    hl_entropy = high_low_range.rolling(window=5).apply(
        lambda x: -np.sum((x / x.sum()) * np.log((x / x.sum()) + 1e-8)) if x.sum() > 0 else 0
    )
    hl_divergence = hl_entropy - hl_entropy.shift(5)
    
    # 3-day close-open fragmentation gradient
    co_fragmentation = (df['close'] - df['open']).abs() / (df['high'] - df['low'] + 1e-8)
    co_gradient = co_fragmentation.rolling(window=3).apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) if len(x) == 3 else 0
    )
    
    # Volume-entropy coupling
    volume_ma = df['volume'].rolling(window=5).mean()
    high_entropy_regime = (hl_entropy > hl_entropy.rolling(window=20).mean()).astype(int)
    volume_fragmentation = df['volume'] / (volume_ma + 1e-8) * high_entropy_regime
    
    # Volume persistence in fragmented microstructure
    volume_persistence = (df['volume'] > df['volume'].shift(1)).rolling(window=3).sum()
    
    # Entropy fragmentation signal
    directional_entropy = hl_divergence * co_gradient
    entropy_signal = directional_entropy * volume_fragmentation * volume_persistence
    
    # Spectral Gap Resonance Patterns
    # 2-day gap resonance spectral density
    gap_resonance = (df['open'] - df['close'].shift(1)).abs()
    spectral_density = gap_resonance.rolling(window=2).std() / (gap_resonance.rolling(window=10).std() + 1e-8)
    
    # 3-day gap phase coherence structure
    gap_phase = np.arctan2(df['open'] - df['close'].shift(1), df['high'] - df['low'])
    phase_coherence = gap_phase.rolling(window=3).apply(
        lambda x: np.abs(np.sum(np.exp(1j * x))) / len(x) if len(x) == 3 else 0
    )
    
    # Resonance efficiency and persistence
    resonance_bandwidth = spectral_density / (spectral_density.rolling(window=10).std() + 1e-8)
    resonance_persistence = (spectral_density > spectral_density.shift(1)).rolling(window=5).sum()
    
    # Spectral resonance signal
    spectral_signal = spectral_density * phase_coherence * resonance_bandwidth * resonance_persistence
    
    # Thermodynamic Microstructure Equilibrium
    # Price-volume energy dissipation
    price_energy = (df['high'] - df['low']).abs() * df['volume']
    energy_dissipation = price_energy - price_energy.rolling(window=5).mean()
    
    # OHLC thermodynamic potential gradients
    ohlc_potential = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    potential_gradient = ohlc_potential.diff(3)
    
    # Equilibrium transitions and phase changes
    regime_change = (energy_dissipation > energy_dissipation.rolling(window=10).std()).astype(int)
    relaxation_pattern = regime_change.rolling(window=5).sum()
    
    # Thermodynamic signal
    thermodynamic_signal = energy_dissipation * potential_gradient * regime_change * relaxation_pattern
    
    # Combine all signals with appropriate weights
    combined_signal = (
        0.4 * entropy_signal.fillna(0) +
        0.3 * spectral_signal.fillna(0) +
        0.3 * thermodynamic_signal.fillna(0)
    )
    
    # Normalize the final signal
    signal_normalized = (combined_signal - combined_signal.rolling(window=20).mean()) / (combined_signal.rolling(window=20).std() + 1e-8)
    
    return signal_normalized
