import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Timeframe Momentum Fracture Detection
    # Short-term momentum fracture
    short_term_momentum = (data['close'] - data['close'].shift(1)) / (data['close'].shift(1) - data['close'].shift(2))
    short_term_fracture = np.abs(short_term_momentum) > 2.0
    
    # Medium-term momentum alignment
    short_return_sign = np.sign(data['close'] - data['close'].shift(1))
    medium_return_sign = np.sign(data['close'] - data['close'].shift(5))
    medium_term_misalignment = short_return_sign != medium_return_sign
    
    # Long-term momentum persistence
    returns_10d = data['close'].pct_change()
    momentum_signs = np.sign(returns_10d.rolling(window=10, min_periods=1).apply(
        lambda x: len(set(x.dropna())), raw=False
    ))
    long_term_persistence = momentum_signs == 1
    
    # Multi-scale fracture
    multi_scale_fracture = short_term_fracture & medium_term_misalignment
    
    # Volume-Fractal Synchronization Analysis
    # Volume acceleration
    volume_acceleration = data['volume'] / data['volume'].shift(1) - 1
    
    # Fractal efficiency
    price_range_10d = np.abs(data['close'] - data['close'].shift(10))
    sum_abs_returns = np.abs(data['close'].diff()).rolling(window=10, min_periods=1).sum()
    fractal_efficiency = price_range_10d / (sum_abs_returns + 1e-8)
    
    # Volume-fractal sync
    volume_fractal_sync = volume_acceleration * fractal_efficiency
    
    # Volume inertia
    volume_ma_5d = data['volume'].rolling(window=5, min_periods=1).mean()
    volume_inertia = data['volume'] / (volume_ma_5d + 1e-8)
    
    # Inertia Barrier Integration
    # Price range compression
    daily_range = data['high'] - data['low']
    range_ma_5d = daily_range.rolling(window=5, min_periods=1).mean()
    price_range_compression = daily_range / (range_ma_5d + 1e-8)
    
    # Volume inertia divergence
    price_acceleration = data['close'].pct_change()
    volume_inertia_divergence = np.sign(volume_inertia - 1) != np.sign(price_acceleration)
    
    # Inertia fracture
    inertia_fracture = multi_scale_fracture & (volume_inertia < 0.6)
    
    # Divergence Confirmation System
    # Momentum-fractal divergence
    momentum_fractal_divergence = multi_scale_fracture.astype(float) * fractal_efficiency
    
    # Volume-inertia divergence
    volume_inertia_divergence_score = volume_inertia_divergence.astype(float) * np.abs(price_acceleration)
    
    # Range-adjusted divergence
    range_adjusted_divergence = momentum_fractal_divergence / (price_range_compression + 1e-8)
    
    # Regime-Based Signal Synthesis
    # High efficiency breakthrough
    high_efficiency_breakthrough = (
        multi_scale_fracture & 
        (fractal_efficiency > fractal_efficiency.rolling(window=20, min_periods=1).quantile(0.7)) &
        (volume_acceleration > 0)
    )
    
    # Inertia collapse reversal
    inertia_collapse_reversal = (
        volume_inertia_divergence & 
        (price_range_compression < 0.5)
    )
    
    # Multi-scale synchronization
    multi_scale_synchronization = (
        multi_scale_fracture & 
        (volume_fractal_sync > volume_fractal_sync.rolling(window=20, min_periods=1).quantile(0.6)) &
        long_term_persistence
    )
    
    # Final factor synthesis
    factor = (
        high_efficiency_breakthrough.astype(float) * 0.4 +
        inertia_collapse_reversal.astype(float) * 0.3 +
        multi_scale_synchronization.astype(float) * 0.3 +
        range_adjusted_divergence * 0.2 +
        volume_inertia_divergence_score * 0.1
    )
    
    return factor
