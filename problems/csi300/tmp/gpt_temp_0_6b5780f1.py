import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Compute Multi-Scale Momentum Decay
    # Short-term momentum decay
    momentum_2d = data['close'] / data['close'].shift(2) - 1
    momentum_5d = data['close'] / data['close'].shift(5) - 1
    momentum_decay_rate = momentum_2d.rolling(window=5).std() / (momentum_5d.rolling(window=5).std() + 1e-8)
    momentum_acceleration = momentum_2d.diff(3).rolling(window=5).mean()
    
    # Medium-term momentum persistence
    momentum_10d = data['close'] / data['close'].shift(10) - 1
    momentum_20d = data['close'] / data['close'].shift(20) - 1
    momentum_stability = momentum_10d.rolling(window=10).std() / (momentum_10d.abs().rolling(window=10).mean() + 1e-8)
    mean_reversion_tendency = -momentum_20d.rolling(window=10).corr(momentum_20d.shift(5))
    
    # Momentum Decay Ratio
    medium_term_persistence = momentum_stability * (1 - mean_reversion_tendency)
    momentum_decay_ratio = momentum_decay_rate / (medium_term_persistence + 1e-8)
    
    # 2. Analyze Volume Clustering Patterns
    # Volume concentration metrics
    volume_3d = data['volume'].rolling(window=3)
    volume_clustering_intensity = volume_3d.std() / (volume_3d.mean() + 1e-8)
    
    volume_8d = data['volume'].rolling(window=8)
    volume_dispersion_ratio = volume_8d.std() / (volume_8d.mean() + 1e-8)
    
    # Volume concentration score (inverse of dispersion)
    volume_concentration = 1 / (volume_dispersion_ratio + 1e-8)
    
    # Detect Volume-Momentum Divergence
    momentum_5d_rolling = momentum_5d.rolling(window=5).mean()
    volume_5d_rolling = data['volume'].rolling(window=5).mean()
    
    # High-volume low-momentum periods
    high_volume_low_momentum = ((volume_5d_rolling > volume_5d_rolling.rolling(window=10).quantile(0.7)) & 
                               (momentum_5d_rolling < momentum_5d_rolling.rolling(window=10).quantile(0.3))).astype(int)
    
    # Low-volume high-momentum periods
    low_volume_high_momentum = ((volume_5d_rolling < volume_5d_rolling.rolling(window=10).quantile(0.3)) & 
                               (momentum_5d_rolling > momentum_5d_rolling.rolling(window=10).quantile(0.7))).astype(int)
    
    # Volume Clustering Divergence Score
    momentum_divergence = high_volume_low_momentum - low_volume_high_momentum
    volume_clustering_divergence = volume_concentration * momentum_divergence
    
    # Apply decay-adjusted weighting
    decay_adjusted_volume_clustering = volume_clustering_divergence * (1 + momentum_decay_ratio)
    
    # 3. Combine with Price Efficiency Filter
    # Measure Price Path Efficiency
    price_path_length = (data['high'] - data['low']).abs().rolling(window=5).sum()
    straight_line_distance = (data['close'] - data['close'].shift(5)).abs()
    price_efficiency = straight_line_distance / (price_path_length + 1e-8)
    
    # Directional consistency ratio
    price_changes = data['close'].diff()
    directional_consistency = (price_changes.rolling(window=5).apply(lambda x: (x > 0).sum() / len(x)) - 0.5).abs()
    
    # Combined price efficiency score
    price_efficiency_score = price_efficiency * (1 - directional_consistency)
    
    # Efficiency-Volume Integration
    efficiency_volume_integration = price_efficiency_score * volume_clustering_intensity
    
    # Final factor with momentum decay adjustment
    final_factor = decay_adjusted_volume_clustering * efficiency_volume_integration
    
    return final_factor
