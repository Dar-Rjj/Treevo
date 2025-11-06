import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    """
    Regime-Aware Multi-Timeframe Alpha Factor
    Combines momentum alignment, volume acceleration, and volatility regime scaling
    """
    # Copy data to avoid modifying original
    data = df.copy()
    
    # Multi-Timeframe Momentum Alignment
    # Short-term momentum (1-3 days)
    mom_1d = data['close'].pct_change(1)
    mom_3d = data['close'].pct_change(3)
    short_term_mom = 0.7 * mom_1d + 0.3 * mom_3d
    
    # Medium-term momentum (5-10 days)
    mom_5d = data['close'].pct_change(5)
    mom_10d = data['close'].pct_change(10)
    # Exponential decay weighting
    weights = np.exp(-np.array([0, 5]) / 7.5)  # Half-life of 7.5 days
    weights = weights / weights.sum()
    medium_term_mom = weights[0] * mom_5d + weights[1] * mom_10d
    
    # Momentum convergence/divergence
    momentum_alignment = np.sign(short_term_mom) * np.sign(medium_term_mom)
    momentum_strength = (abs(short_term_mom) + abs(medium_term_mom)) / 2
    aligned_momentum = momentum_alignment * momentum_strength
    
    # Volume Acceleration Confirmation
    def calculate_volume_slope(volume_series):
        """Calculate linear regression slope of volume series"""
        if len(volume_series) < 2:
            return 0
        x = np.arange(len(volume_series))
        slope, _, _, _, _ = stats.linregress(x, volume_series)
        return slope
    
    # Recent volume slope (last 5 days)
    recent_slope = data['volume'].rolling(window=5).apply(
        calculate_volume_slope, raw=False
    )
    
    # Historical volume slope (days 6-15 ago)
    historical_slope = data['volume'].shift(5).rolling(window=10).apply(
        calculate_volume_slope, raw=False
    )
    
    volume_acceleration = recent_slope - historical_slope
    
    # Volume-momentum interaction
    volume_weight = 1 + np.tanh(volume_acceleration * 10)  # Scale acceleration
    momentum_with_volume = aligned_momentum * volume_weight
    
    # Volatility-Regime Scaling
    # Range-based volatility
    daily_volatility = (data['high'] - data['low']) / data['close']
    
    # Rolling volatility percentiles
    vol_20d = daily_volatility.rolling(window=20)
    vol_80th = vol_20d.quantile(0.8)
    vol_20th = vol_20d.quantile(0.2)
    
    # Volatility regime classification
    high_vol_regime = daily_volatility > vol_80th
    low_vol_regime = daily_volatility < vol_20th
    normal_vol_regime = ~high_vol_regime & ~low_vol_regime
    
    # Regime-specific scaling factors
    vol_scaling = pd.Series(1.0, index=data.index)
    vol_scaling[high_vol_regime] = 0.5  # Reduce position in high volatility
    vol_scaling[low_vol_regime] = 1.5   # Increase position in low volatility
    # Normal volatility remains at 1.0
    
    # Additional confirmation for low volatility regime
    low_vol_confirm = (momentum_alignment > 0) & (volume_acceleration > 0)
    vol_scaling[low_vol_regime & ~low_vol_confirm] = 1.0
    
    # Composite Alpha Generation
    alpha_factor = momentum_with_volume * vol_scaling
    
    # Clean and normalize
    alpha_factor = alpha_factor.replace([np.inf, -np.inf], np.nan)
    alpha_factor = alpha_factor.fillna(0)
    
    # Z-score normalization
    alpha_factor = (alpha_factor - alpha_factor.rolling(window=20).mean()) / alpha_factor.rolling(window=20).std()
    alpha_factor = alpha_factor.fillna(0)
    
    return alpha_factor
