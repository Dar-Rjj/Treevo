import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Normalized Multi-Timeframe Momentum with Volume Divergence
    """
    # Calculate returns for momentum calculations
    returns = df['close'].pct_change()
    
    # Multi-Timeframe Momentum Calculation
    # Short-term momentum (3-day)
    mom_3d = df['close'] / df['close'].shift(3) - 1
    
    # Medium-term momentum (10-day)
    mom_10d = df['close'] / df['close'].shift(10) - 1
    
    # Long-term momentum (20-day)
    mom_20d = df['close'] / df['close'].shift(20) - 1
    
    # Volatility Normalization
    # Calculate rolling volatility (20-day standard deviation)
    volatility = returns.rolling(window=20).std()
    
    # Normalize each momentum component by volatility
    # Add small epsilon to avoid division by zero
    epsilon = 1e-8
    mom_3d_norm = mom_3d / (volatility + epsilon)
    mom_10d_norm = mom_10d / (volatility + epsilon)
    mom_20d_norm = mom_20d / (volatility + epsilon)
    
    # Volume Divergence Detection
    # Calculate 5-day price trend (slope)
    price_trend = df['close'].rolling(window=5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False
    )
    
    # Calculate 5-day volume trend (slope)
    volume_trend = df['volume'].rolling(window=5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False
    )
    
    # Calculate divergence signals
    bullish_divergence = ((price_trend > 0) & (volume_trend < 0)).astype(float)
    bearish_divergence = ((price_trend < 0) & (volume_trend > 0)).astype(float)
    
    # Convergence strength (when price and volume move together)
    convergence_strength = np.sign(price_trend) * np.sign(volume_trend) * (
        np.abs(price_trend) * np.abs(volume_trend)
    )
    
    # Combined volume divergence signal
    volume_signal = bullish_divergence - bearish_divergence + 0.5 * convergence_strength
    
    # Regime-Based Weighting
    # Volatility regime classification
    vol_60d_median = volatility.rolling(window=60).median()
    high_vol_regime = (volatility > vol_60d_median).astype(float)
    low_vol_regime = (volatility <= vol_60d_median).astype(float)
    
    # Adaptive component weights based on regime
    # High volatility: emphasize short-term momentum
    short_weight = high_vol_regime * 0.6 + low_vol_regime * 0.2
    medium_weight = high_vol_regime * 0.3 + low_vol_regime * 0.3
    long_weight = high_vol_regime * 0.1 + low_vol_regime * 0.5
    
    # Volume divergence weight constant
    volume_weight = 1.0
    
    # Final Alpha Factor Construction
    # Combine normalized momentum components with regime-based weights
    momentum_composite = (
        short_weight * mom_3d_norm +
        medium_weight * mom_10d_norm +
        long_weight * mom_20d_norm
    )
    
    # Multiply by volume divergence signal
    alpha_factor = momentum_composite * volume_signal * volume_weight
    
    # Clean and return the factor
    alpha_factor = alpha_factor.replace([np.inf, -np.inf], np.nan)
    alpha_factor = alpha_factor.fillna(0)
    
    return alpha_factor
