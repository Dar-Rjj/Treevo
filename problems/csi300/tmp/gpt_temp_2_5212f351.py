import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Volume-Confirmed Momentum Factor that combines short-term momentum with volume-based confirmation
    and long-term trend consistency.
    """
    close = data['close']
    high = data['high']
    low = data['low']
    volume = data['volume']
    
    # Initialize result series
    factor = pd.Series(index=close.index, dtype=float)
    
    # Short-Term Momentum Component
    # 5-day Exponential Weighted Price Momentum with 3-day half-life
    alpha_5d = 1 - np.exp(-np.log(2)/3)
    momentum_5d = close.ewm(alpha=alpha_5d).mean().pct_change(periods=4)
    
    # 10-day Exponential Weighted Price Momentum with 5-day half-life
    alpha_10d = 1 - np.exp(-np.log(2)/5)
    momentum_10d = close.ewm(alpha=alpha_10d).mean().pct_change(periods=9)
    
    # Combined short-term momentum
    short_term_momentum = 0.6 * momentum_5d + 0.4 * momentum_10d
    
    # Volatility scaling using high-low range
    high_low_range = (high - low) / ((high + low) / 2)
    volatility_10d = high_low_range.rolling(window=10).std()
    
    # Volatility-scaled short-term momentum (preserving sign)
    volatility_scaled_momentum = short_term_momentum / (volatility_10d + 1e-8)
    
    # Long-Term Confirmation Component
    # 20-day Volume-Weighted Price Trend
    price_change_20d = close.pct_change()
    volume_weighted_20d = (price_change_20d * volume).rolling(window=20).sum() / volume.rolling(window=20).sum()
    
    # 50-day Volume-Weighted Price Trend
    volume_weighted_50d = (price_change_20d * volume).rolling(window=50).sum() / volume.rolling(window=50).sum()
    
    # Long-Term Trend Consistency
    trend_consistency = np.sign(volume_weighted_20d) * np.sign(volume_weighted_50d)
    trend_strength = (np.abs(volume_weighted_20d) + np.abs(volume_weighted_50d)) / 2
    
    # Volume Confirmation Analysis
    # Volume-Price Divergence Detection
    volume_trend_10d = volume.rolling(window=10).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True)
    price_trend_10d = close.rolling(window=10).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True)
    
    volume_price_divergence = np.sign(volume_trend_10d) != np.sign(price_trend_10d)
    divergence_strength = np.abs(volume_trend_10d * price_trend_10d)
    
    # Volume Acceleration Confirmation
    volume_5d_growth = volume.rolling(window=5).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True)
    volume_10d_growth = volume.rolling(window=10).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True)
    volume_acceleration = volume_5d_growth - volume_10d_growth
    
    volume_confirmation = np.sign(volume_acceleration) == np.sign(short_term_momentum)
    acceleration_strength = np.abs(volume_acceleration * short_term_momentum)
    
    # Volume Stability Assessment
    volume_volatility_20d = volume.pct_change().rolling(window=20).std()
    volume_stability_penalty = volume_volatility_20d * (1 - np.abs(short_term_momentum))
    
    # Factor Combination & Robustness
    # Volume confirmation strength
    volume_confirmation_strength = acceleration_strength.rank(pct=True) - volume_stability_penalty.rank(pct=True)
    
    # Multi-Timeframe Momentum Blending
    base_factor = volatility_scaled_momentum * (1 + 0.5 * volume_confirmation_strength)
    
    # Apply trend consistency scaling
    trend_adjusted_factor = base_factor * (1 + 0.3 * trend_consistency * trend_strength.rank(pct=True))
    
    # Divergence Penalty Application
    divergence_penalty = volume_price_divergence.astype(float) * divergence_strength.rank(pct=True)
    persistent_divergence = volume_price_divergence.rolling(window=5).sum() / 5
    
    # Final factor with divergence penalties
    final_factor = trend_adjusted_factor * (1 - 0.4 * divergence_penalty - 0.3 * persistent_divergence)
    
    # Robustness Transformations
    # Rank-based transformation to handle outliers
    ranked_factor = final_factor.rank(pct=True) - 0.5
    
    # Bounded factor values for stability
    bounded_factor = np.tanh(ranked_factor * 3)  # Scale and bound between -1 and 1
    
    return bounded_factor
