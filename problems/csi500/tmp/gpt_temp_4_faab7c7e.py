import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Price Entropy Measurement
    # Intraday Entropy
    high_low_range = data['high'] - data['low']
    high_open_ratio = (data['high'] - data['open']) / high_low_range
    open_low_ratio = (data['open'] - data['low']) / high_low_range
    
    # Handle division by zero and invalid values
    high_open_ratio = high_open_ratio.replace([np.inf, -np.inf], np.nan).fillna(0.5)
    open_low_ratio = open_low_ratio.replace([np.inf, -np.inf], np.nan).fillna(0.5)
    
    # Calculate intraday entropy
    intraday_entropy = -(
        high_open_ratio * np.log(np.maximum(high_open_ratio, 1e-10)) + 
        open_low_ratio * np.log(np.maximum(open_low_ratio, 1e-10))
    )
    
    # Entropy Persistence
    entropy_diff_sign = np.sign(intraday_entropy.diff())
    entropy_persistence = (
        entropy_diff_sign.rolling(window=5, min_periods=1)
        .apply(lambda x: (x == x.iloc[-1]).sum() if len(x) > 0 else 0, raw=False)
    )
    
    # Multi-day Entropy Dynamics
    entropy_3d_vector = pd.concat([
        intraday_entropy,
        intraday_entropy.shift(1),
        intraday_entropy.shift(2)
    ], axis=1)
    entropy_gradient = intraday_entropy - intraday_entropy.shift(2)
    
    # Volume Entropy Framework
    # Volume Distribution Entropy
    volume_4d_sum = data['volume'].rolling(window=5, min_periods=1).sum()
    volume_concentration = data['volume'] / volume_4d_sum
    volume_concentration = volume_concentration.replace([np.inf, -np.inf], np.nan).fillna(0.2)
    
    volume_entropy = -volume_concentration * np.log(np.maximum(volume_concentration, 1e-10))
    
    # Volume-Entropy Momentum
    volume_entropy_change = volume_entropy - volume_entropy.shift(3)
    volume_entropy_acceleration = volume_entropy_change - volume_entropy_change.shift(1)
    
    # Multi-Scale Momentum Divergence
    # Temporal Momentum Structure
    short_momentum = data['close'] / data['close'].shift(2) - 1
    medium_momentum = data['close'] / data['close'].shift(7) - 1
    
    # Momentum Divergence Matrix
    momentum_spread = medium_momentum - short_momentum
    momentum_curvature = (
        (short_momentum - short_momentum.shift(2)) - 
        (medium_momentum - medium_momentum.shift(2))
    )
    
    # Entropy-Momentum Coupling
    entropy_direction = np.sign(entropy_gradient)
    momentum_direction = np.sign(medium_momentum)
    alignment_score = entropy_direction * momentum_direction
    
    coupling_persistence = (
        (alignment_score > 0).rolling(window=5, min_periods=1)
        .apply(lambda x: x.sum(), raw=False)
    )
    
    # Microstructural Entropic Patterns
    # Price-Volume Entropic Synchronization
    pv_entropy_correlation = np.sign(entropy_gradient) * np.sign(volume_entropy_change)
    synchronization_strength = np.abs(entropy_gradient) * np.abs(volume_entropy_change)
    
    consistent_sync_days = (
        (pv_entropy_correlation > 0).rolling(window=5, min_periods=1)
        .apply(lambda x: x.sum(), raw=False)
    )
    sync_stability = synchronization_strength.rolling(window=5, min_periods=1).std()
    
    # Entropic Microstructure Quality
    entropy_range_ratio = intraday_entropy / (np.log(2) * (data['high'] - data['low']) / data['close'])
    entropy_range_ratio = entropy_range_ratio.replace([np.inf, -np.inf], np.nan).fillna(0.5)
    
    efficiency_persistence = (
        (entropy_range_ratio > 0.5).rolling(window=5, min_periods=1)
        .apply(lambda x: x.sum(), raw=False)
    )
    
    volume_entropy_stability = volume_entropy / volume_entropy.shift(3)
    volume_entropy_stability = volume_entropy_stability.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    
    volume_entropy_compression = volume_entropy_change / volume_entropy_acceleration
    volume_entropy_compression = volume_entropy_compression.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    # Divergence State Construction
    # Base Entropic State Scoring
    entropic_state_score = pd.Series(0.0, index=data.index)
    
    # High Entropy Growth
    high_entropy_growth_mask = (entropy_gradient > 0) & (alignment_score > 0)
    entropic_state_score[high_entropy_growth_mask] = 2.0
    
    # Low Entropy Compression
    low_entropy_compression_mask = (entropy_gradient < 0) & (alignment_score > 0)
    entropic_state_score[low_entropy_compression_mask] = 1.5
    
    # Entropic Divergence
    entropic_divergence_mask = (alignment_score < 0)
    entropic_state_score[entropic_divergence_mask] = -1.0
    
    # Neutral Entropy
    neutral_entropy_mask = (np.abs(entropy_gradient) < 0.01)
    entropic_state_score[neutral_entropy_mask] = 0.8
    
    # Momentum Divergence Integration
    momentum_divergence_component = (
        momentum_spread * entropy_gradient + 
        momentum_curvature * volume_entropy_change
    )
    
    # Microstructure Enhancement
    synchronization_multiplier = pd.Series(1.0, index=data.index)
    
    # Strong Synchronization
    strong_sync_mask = (consistent_sync_days > 3)
    synchronization_multiplier[strong_sync_mask] *= 1.4
    
    # High Efficiency
    high_efficiency_mask = (efficiency_persistence > 3)
    synchronization_multiplier[high_efficiency_mask] *= 1.3
    
    # Stable Volume Entropy
    stable_volume_mask = (volume_entropy_stability > 0.9)
    synchronization_multiplier[stable_volume_mask] *= 1.2
    
    # Quality Filters
    entropic_efficiency_filter = entropy_range_ratio * efficiency_persistence
    volume_quality_filter = volume_entropy_stability * volume_entropy_compression
    
    # Temporal Scale Adaptation
    # Short-Scale Emphasis
    short_scale_component = entropic_state_score * short_momentum * 1.5
    
    # Medium-Scale Integration
    medium_scale_component = (
        entropic_state_score * momentum_spread * entropy_gradient * 
        synchronization_multiplier
    )
    
    # Multi-Scale Coherence
    scale_alignment = np.sign(short_momentum) * np.sign(medium_momentum)
    coherence_strength = (
        scale_alignment * entropic_efficiency_filter * volume_quality_filter
    )
    
    # Final Factor Construction
    base_factor = (
        entropic_state_score + 
        momentum_divergence_component + 
        short_scale_component + 
        medium_scale_component
    )
    
    # Apply quality filters and coherence
    final_factor = (
        base_factor * 
        synchronization_multiplier * 
        entropic_efficiency_filter * 
        volume_quality_filter * 
        coherence_strength
    )
    
    return final_factor.fillna(0)
