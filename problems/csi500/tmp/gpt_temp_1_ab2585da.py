import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    """
    Momentum-Volume Regime Factor that combines short-term momentum signals
    with volume acceleration and volatility scaling, incorporating regime confirmation
    through multi-timeframe alignment and divergence detection.
    """
    # Extract price and volume data
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # Initialize result series
    factor = pd.Series(index=df.index, dtype=float)
    
    # Short-Term Momentum Component
    momentum_5d = (close / close.shift(5) - 1).fillna(0)
    momentum_10d = (close / close.shift(10) - 1).fillna(0)
    
    # Volume Acceleration Component
    # 5-day volume ratio: recent 5 days vs previous 5 days
    vol_recent_5d = volume.rolling(window=5).mean()
    vol_previous_5d = volume.shift(5).rolling(window=5).mean()
    volume_ratio = (vol_recent_5d / vol_previous_5d - 1).fillna(0)
    
    # Volume trend using linear regression slope
    def volume_slope(vol_series):
        if len(vol_series) < 2:
            return 0
        x = np.arange(len(vol_series))
        slope, _, _, _, _ = stats.linregress(x, vol_series)
        return slope
    
    volume_trend = volume.rolling(window=10).apply(volume_slope, raw=False).fillna(0)
    
    # Volatility Scaling
    # Calculate 20-day volatility using high-low range
    daily_range = (high - low) / close
    volatility_20d = daily_range.rolling(window=20).std().fillna(0.01)  # Avoid division by zero
    
    # Scale momentum by volatility
    momentum_5d_scaled = momentum_5d / (volatility_20d + 1e-8)
    momentum_10d_scaled = momentum_10d / (volatility_20d + 1e-8)
    
    # Regime Confirmation
    # Multi-Timeframe Alignment
    alignment_strength = np.where(
        np.sign(momentum_5d) == np.sign(momentum_10d),
        np.abs(momentum_5d) + np.abs(momentum_10d),
        (np.abs(momentum_5d) + np.abs(momentum_10d)) * 0.5  # Reduce weight when not aligned
    )
    
    # Volume-Momentum Divergence Detection
    momentum_direction = np.sign(momentum_5d)
    volume_direction = np.sign(volume_trend)
    
    divergence_adjustment = np.where(
        momentum_direction != volume_direction,
        -0.3,  # Penalty for divergence
        1.0    # No adjustment for convergence
    )
    
    # Combine all components
    base_factor = (momentum_5d_scaled * 0.6 + momentum_10d_scaled * 0.4) * alignment_strength
    volume_component = volume_ratio * 0.3 + volume_trend * 0.2
    
    # Final factor with divergence adjustment
    factor = (base_factor + volume_component) * divergence_adjustment
    
    return factor.fillna(0)
