import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price-Volume Momentum Convergence with Adaptive Confirmation alpha factor
    """
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Ensure sufficient data length
    if len(df) < 20:
        return result
    
    # Extract price and volume data
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # 1. Momentum Divergence Detection
    # Short-term momentum (3-day)
    short_momentum = (close - close.shift(3)) / close.shift(3)
    short_momentum_sign = np.sign(short_momentum)
    
    # Medium-term momentum (10-day)
    medium_momentum = (close - close.shift(10)) / close.shift(10)
    medium_momentum_sign = np.sign(medium_momentum)
    
    # Momentum divergence score
    momentum_divergence = short_momentum - medium_momentum
    divergence_magnitude = np.abs(momentum_divergence)
    sign_agreement = (short_momentum_sign == medium_momentum_sign).astype(float)
    momentum_div_score = momentum_divergence * divergence_magnitude * sign_agreement
    
    # 2. Volume Confirmation Framework
    # Volume acceleration
    volume_acceleration = (volume - volume.shift(1)) / volume.shift(1)
    volume_acceleration_sign = np.sign(volume_acceleration)
    
    # Volume persistence (5-day rolling trend consistency)
    volume_change_sign = np.sign(volume.diff())
    volume_persistence = volume_change_sign.rolling(window=5).apply(
        lambda x: len(set(x.dropna())) == 1 if len(x.dropna()) == 5 else 0, raw=False
    )
    volume_persistence = volume_persistence * np.abs(volume_acceleration)
    
    # Volume-momentum alignment
    volume_momentum_alignment = volume_acceleration_sign * short_momentum_sign
    volume_confirmation = volume_momentum_alignment * volume_persistence
    
    # 3. Regime-Adaptive Signal Scaling
    # Volatility context (10-day price range average)
    daily_range = (high - low) / close
    volatility_context = daily_range.rolling(window=10).mean()
    
    # Relative volume strength
    volume_20ma = volume.rolling(window=20).mean()
    relative_volume_strength = volume / volume_20ma
    
    # Adaptive scaling
    volatility_adjusted_momentum = momentum_div_score / (volatility_context + 1e-8)
    scaled_momentum = volatility_adjusted_momentum * relative_volume_strength
    
    # Volume confirmation as binary filter
    volume_filter = (volume_confirmation > 0).astype(float)
    
    # 4. Persistence Enhancement
    # Momentum direction consistency (5-day)
    momentum_direction = np.sign(close.diff())
    momentum_consistency = momentum_direction.rolling(window=5).apply(
        lambda x: len(set(x.dropna())) == 1 if len(x.dropna()) == 5 else 0, raw=False
    )
    
    # Volume trend alignment duration
    volume_trend_alignment = (volume_acceleration_sign == short_momentum_sign).astype(int)
    volume_alignment_duration = volume_trend_alignment.rolling(window=5).sum() / 5
    
    # Persistence multiplier
    persistence_multiplier = momentum_consistency * volume_alignment_duration
    
    # 5. Composite Alpha Factor
    # Core momentum component with volume confirmation
    core_component = scaled_momentum * volume_filter
    
    # Apply persistence enhancement
    final_alpha = core_component * persistence_multiplier
    
    # Preserve original sign and magnitude
    alpha_sign = np.sign(momentum_div_score)
    final_alpha = final_alpha * alpha_sign
    
    # Handle NaN values
    result = final_alpha.fillna(0)
    
    return result
