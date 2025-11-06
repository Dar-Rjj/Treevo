import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Nonlinear Volume-Price Acceleration Divergence factor
    Combines price acceleration dynamics with volume acceleration patterns
    to detect sustainable divergence signals
    """
    data = df.copy()
    
    # Price acceleration dynamics
    # Second-order price momentum (rate of change of momentum)
    price_momentum_3d = data['close'].pct_change(periods=3)
    price_momentum_5d = data['close'].pct_change(periods=5)
    price_momentum_8d = data['close'].pct_change(periods=8)
    
    # Second-order acceleration
    price_accel_3d = price_momentum_3d.diff(3)
    price_accel_5d = price_momentum_5d.diff(5)
    price_accel_8d = price_momentum_8d.diff(8)
    
    # Acceleration persistence across windows
    accel_persistence = (
        price_accel_3d.rolling(window=5, min_periods=3).mean() +
        price_accel_5d.rolling(window=5, min_periods=3).mean() +
        price_accel_8d.rolling(window=5, min_periods=3).mean()
    ) / 3
    
    # Detect acceleration regime shifts using inflection points
    accel_trend_3d = price_accel_3d.rolling(window=5, min_periods=3).apply(
        lambda x: 1 if (x.iloc[-1] > x.iloc[0] and x.iloc[-1] > 0) else (-1 if (x.iloc[-1] < x.iloc[0] and x.iloc[-1] < 0) else 0)
    )
    accel_trend_5d = price_accel_5d.rolling(window=5, min_periods=3).apply(
        lambda x: 1 if (x.iloc[-1] > x.iloc[0] and x.iloc[-1] > 0) else (-1 if (x.iloc[-1] < x.iloc[0] and x.iloc[-1] < 0) else 0)
    )
    regime_shift_strength = (accel_trend_3d + accel_trend_5d) / 2
    
    # Volume acceleration patterns
    volume_momentum_3d = data['volume'].pct_change(periods=3)
    volume_momentum_5d = data['volume'].pct_change(periods=5)
    volume_momentum_8d = data['volume'].pct_change(periods=8)
    
    # Volume acceleration
    volume_accel_3d = volume_momentum_3d.diff(3)
    volume_accel_5d = volume_momentum_5d.diff(5)
    volume_accel_8d = volume_momentum_8d.diff(8)
    
    # Volume acceleration consistency
    volume_accel_consistency = (
        volume_accel_3d.rolling(window=5, min_periods=3).std() +
        volume_accel_5d.rolling(window=5, min_periods=3).std() +
        volume_accel_8d.rolling(window=5, min_periods=3).std()
    ) / 3
    
    # Identify volume acceleration clusters using rolling z-score
    volume_accel_zscore = volume_accel_5d.rolling(window=20, min_periods=10).apply(
        lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() > 0 else 0
    )
    volume_clusters = volume_accel_zscore.rolling(window=5, min_periods=3).apply(
        lambda x: 1 if abs(x.mean()) > 1 else 0
    )
    
    # Divergence signal construction
    # Compare price vs volume acceleration trajectories
    price_volume_correlation = price_accel_5d.rolling(window=10, min_periods=5).corr(volume_accel_5d)
    
    # Directional discrepancy measures
    price_direction = np.sign(price_accel_5d)
    volume_direction = np.sign(volume_accel_5d)
    directional_discrepancy = (price_direction != volume_direction).astype(int) * abs(price_accel_5d - volume_accel_5d)
    
    # Persistence filters for sustainable divergence
    divergence_persistence = directional_discrepancy.rolling(window=5, min_periods=3).mean()
    
    # Final factor construction
    # Combine acceleration dynamics with volume patterns and divergence signals
    factor = (
        accel_persistence * 0.3 +
        regime_shift_strength * 0.2 +
        volume_accel_consistency * 0.15 +
        volume_clusters * 0.15 +
        (1 - price_volume_correlation) * 0.1 +
        divergence_persistence * 0.1
    )
    
    # Normalize the factor
    factor = (factor - factor.rolling(window=50, min_periods=30).mean()) / factor.rolling(window=50, min_periods=30).std()
    
    return factor
