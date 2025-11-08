import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Normalized Multi-Timeframe Momentum with Volume Divergence
    
    Parameters:
    df: DataFrame with columns ['open', 'high', 'low', 'close', 'amount', 'volume']
    
    Returns:
    Series: Composite alpha factor values
    """
    
    # Multi-Timeframe Momentum Calculation
    close = df['close']
    
    # Short-term momentum (3-day)
    mom_3d = close.pct_change(3)
    
    # Medium-term momentum (5-day)
    mom_5d = close.pct_change(5)
    
    # Long-term momentum (10-day)
    mom_10d = close.pct_change(10)
    
    # Volatility Normalization
    # Calculate 20-day rolling volatility based on daily returns
    daily_returns = close.pct_change()
    volatility_20d = daily_returns.rolling(window=20).std()
    
    # Normalize each momentum component by volatility
    # Add small epsilon to avoid division by zero
    epsilon = 1e-8
    norm_mom_3d = mom_3d / (volatility_20d + epsilon)
    norm_mom_5d = mom_5d / (volatility_20d + epsilon)
    norm_mom_10d = mom_10d / (volatility_20d + epsilon)
    
    # Volume Divergence Detection
    volume = df['volume']
    
    # Calculate 5-day price trend slope using linear regression
    def calc_slope(series):
        if len(series) < 5:
            return np.nan
        x = np.arange(len(series))
        return np.polyfit(x, series.values, 1)[0]
    
    price_slope_5d = close.rolling(window=5).apply(calc_slope, raw=False)
    volume_slope_5d = volume.rolling(window=5).apply(calc_slope, raw=False)
    
    # Volume divergence signal: positive when price and volume trends align
    volume_divergence = np.sign(price_slope_5d) * np.sign(volume_slope_5d)
    
    # Volume confirmation strength (absolute value of volume slope)
    volume_strength = np.abs(volume_slope_5d)
    
    # Combined volume signal
    volume_signal = volume_divergence * volume_strength
    
    # Regime-Based Weighting
    # Volatility regime classification
    vol_quantile_high = volatility_20d.quantile(0.7)
    vol_quantile_low = volatility_20d.quantile(0.3)
    
    # Create regime indicators
    high_vol_regime = (volatility_20d > vol_quantile_high).astype(int)
    low_vol_regime = (volatility_20d < vol_quantile_low).astype(int)
    transition_regime = 1 - high_vol_regime - low_vol_regime
    
    # Adaptive component weights
    # Base weights
    base_weights = np.array([0.3, 0.4, 0.3])  # 3d, 5d, 10d
    
    # High volatility: emphasize longer timeframe
    high_vol_weights = np.array([0.2, 0.3, 0.5])
    
    # Low volatility: emphasize shorter timeframe
    low_vol_weights = np.array([0.5, 0.3, 0.2])
    
    # Transition: balanced with volume emphasis
    transition_weights = np.array([0.33, 0.33, 0.34])
    
    # Apply regime-specific weights
    weights_3d = (high_vol_regime * high_vol_weights[0] + 
                  low_vol_regime * low_vol_weights[0] + 
                  transition_regime * transition_weights[0])
    
    weights_5d = (high_vol_regime * high_vol_weights[1] + 
                  low_vol_regime * low_vol_weights[1] + 
                  transition_regime * transition_weights[1])
    
    weights_10d = (high_vol_regime * high_vol_weights[2] + 
                   low_vol_regime * low_vol_weights[2] + 
                   transition_regime * transition_weights[2])
    
    # Composite Alpha Factor
    # Weighted combination of normalized momentums
    weighted_momentum = (weights_3d * norm_mom_3d + 
                         weights_5d * norm_mom_5d + 
                         weights_10d * norm_mom_10d)
    
    # Volume divergence adjustment
    # Stronger adjustment during transition regimes
    volume_adjustment = volume_signal * (1 + 2 * transition_regime)
    
    # Final composite factor
    composite_factor = weighted_momentum * (1 + 0.3 * volume_adjustment)
    
    return composite_factor
