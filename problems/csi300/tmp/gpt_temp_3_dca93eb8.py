import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Novel alpha factor combining entropic dynamics, quantum-inspired amplitude-phase analysis,
    temporal fracture detection, regime switching, microstructural memory, and fractal correlations.
    """
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Required columns
    cols = ['open', 'high', 'low', 'close', 'volume']
    for col in cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Calculate basic price and volume features
    df = df.copy()
    df['price_change'] = df['close'].diff().abs()
    df['volume_change'] = df['volume'].pct_change()
    
    # 1. Entropic Price-Volume Coherence with Microstructural Anchors
    window_entropy = 20
    
    # Price entropy
    price_changes_abs = df['price_change'].rolling(window=window_entropy, min_periods=1).apply(
        lambda x: -np.sum((x/x.sum()) * np.log((x/x.sum()).replace(0, 1e-10))) if x.sum() > 0 else 0, 
        raw=False
    )
    
    # Volume entropy
    volume_entropy = df['volume'].rolling(window=window_entropy, min_periods=1).apply(
        lambda x: -np.sum((x/x.sum()) * np.log((x/x.sum()).replace(0, 1e-10))) if x.sum() > 0 else 0,
        raw=False
    )
    
    # Coherence divergence
    entropy_coherence = (price_changes_abs - volume_entropy).abs()
    
    # Opening gap persistence anchor
    opening_gap = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    gap_persistence = opening_gap.rolling(window=5).std()
    
    # 2. Quantum-Inspired Price Amplitude with Volume Phase Shift
    price_amplitude = (df['high'] - df['low']) / (df['close'] - df['open']).abs().replace(0, 1e-10)
    volume_phase = np.arctan(df['volume_change'].replace([np.inf, -np.inf], 0).fillna(0))
    
    # Amplitude-phase resonance
    amplitude_phase_resonance = price_amplitude * np.cos(volume_phase)
    
    # 3. Temporal Fracture Points with Volume Cascade Dynamics
    fracture_condition = (df['high'] - df['low']) > (2 * (df['close'] - df['open']).abs())
    fracture_points = fracture_condition.astype(int)
    
    # Volume cascade intensity
    min_vol_rolling = df['volume'].rolling(window=3, min_periods=1).min().shift(1)
    volume_cascade = df['volume'] / min_vol_rolling.replace(0, 1e-10)
    
    # Fracture clustering
    fracture_clustering = fracture_points.rolling(window=10, min_periods=1).sum()
    
    # 4. Multi-Dimensional Regime Switching with Entropic Boundaries
    price_efficiency = df['price_change'] / (df['high'] - df['low']).replace(0, 1e-10)
    
    # Volume distribution boundaries
    volume_quantile_25 = df['volume'].rolling(window=50, min_periods=1).quantile(0.25)
    volume_quantile_75 = df['volume'].rolling(window=50, min_periods=1).quantile(0.75)
    
    # Regime detection
    low_vol_regime = (df['volume'] < volume_quantile_25).astype(int)
    high_vol_regime = (df['volume'] > volume_quantile_75).astype(int)
    normal_regime = ((df['volume'] >= volume_quantile_25) & (df['volume'] <= volume_quantile_75)).astype(int)
    
    # 5. Microstructural Memory with Price-Volume Hysteresis
    price_hysteresis = (df['close'] - df['open']) / (df['close'].shift(1) - df['open'].shift(1)).replace(0, 1e-10)
    volume_memory_decay = (df['volume'] / df['volume'].shift(1)) - (df['volume'].shift(1) / df['volume'].shift(2))
    
    # Hysteresis persistence
    hysteresis_persistence = price_hysteresis.rolling(window=5).std()
    
    # 6. Fractal Correlation Dimension Approximation
    # Simplified correlation dimension using price movement clustering
    price_movement_cluster = (df['price_change'] > df['price_change'].rolling(window=20, min_periods=1).mean()).astype(int)
    fractal_dimension = price_movement_cluster.rolling(window=10, min_periods=1).mean()
    
    # Volume attractor points (local maxima in volume/price_change space)
    volume_price_ratio = df['volume'] / df['price_change'].replace(0, 1e-10)
    volume_attractor = (volume_price_ratio > volume_price_ratio.rolling(window=5, center=True, min_periods=1).mean()).astype(int)
    
    # Combine all components with appropriate weights
    alpha_component_1 = entropy_coherence * gap_persistence
    alpha_component_2 = amplitude_phase_resonance
    alpha_component_3 = fracture_clustering * volume_cascade
    alpha_component_4 = price_efficiency * (low_vol_regime - high_vol_regime)
    alpha_component_5 = price_hysteresis * volume_memory_decay / hysteresis_persistence.replace(0, 1e-10)
    alpha_component_6 = fractal_dimension * volume_attractor
    
    # Final alpha combination
    result = (
        0.25 * alpha_component_1 +
        0.20 * alpha_component_2 +
        0.15 * alpha_component_3 +
        0.15 * alpha_component_4 +
        0.15 * alpha_component_5 +
        0.10 * alpha_component_6
    )
    
    # Normalize and handle edge cases
    result = result.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(0)
    
    # Z-score normalization
    result = (result - result.rolling(window=252, min_periods=1).mean()) / result.rolling(window=252, min_periods=1).std().replace(0, 1e-10)
    
    return result
