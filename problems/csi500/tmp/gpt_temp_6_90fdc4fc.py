import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Adaptive Momentum Velocity with Dynamic Volume Confirmation alpha factor
    
    Parameters:
    df: DataFrame with columns ['open', 'high', 'low', 'close', 'amount', 'volume']
    
    Returns:
    Series: Alpha factor values indexed by date
    """
    
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    # Calculate raw velocity with zero range handling
    denominator = df['high'] - df['low']
    denominator = denominator.replace(0, 0.0001)  # Handle zero range
    velocity = (df['close'] - df['open']) / denominator
    
    # Velocity trend analysis
    velocity_direction = np.sign(velocity)
    velocity_magnitude = np.abs(velocity)
    velocity_acceleration = velocity.diff()
    
    # Multi-period velocity persistence
    persistence_count = pd.Series(0, index=df.index)
    persistence_strength = pd.Series(0.0, index=df.index)
    
    for i in range(1, len(df)):
        if velocity_direction.iloc[i] == velocity_direction.iloc[i-1]:
            persistence_count.iloc[i] = persistence_count.iloc[i-1] + 1
            persistence_strength.iloc[i] = persistence_strength.iloc[i-1] + velocity_magnitude.iloc[i]
        else:
            persistence_count.iloc[i] = 1
            persistence_strength.iloc[i] = velocity_magnitude.iloc[i]
    
    # Exponential decay weighting system
    decay_weighted_persistence = pd.Series(0.0, index=df.index)
    
    for i in range(len(df)):
        if i == 0:
            decay_weighted_persistence.iloc[i] = persistence_strength.iloc[i]
        else:
            # Calculate velocity volatility for adaptive decay
            if i >= 5:
                velocity_volatility = velocity.iloc[max(0, i-4):i+1].std()
                if velocity_volatility > velocity.rolling(20).std().iloc[i]:
                    decay_factor = 0.85  # Higher volatility -> faster decay
                else:
                    decay_factor = 0.95  # Lower volatility -> slower decay
            else:
                decay_factor = 0.92  # Default decay factor
            
            decay_weighted_persistence.iloc[i] = (
                decay_factor * decay_weighted_persistence.iloc[i-1] + 
                persistence_strength.iloc[i]
            )
    
    # Dynamic volume confirmation
    volume_change_ratio = df['volume'] / df['volume'].shift(1)
    volume_velocity = np.sign(volume_change_ratio - 1)
    volume_momentum = df['volume'].diff()
    
    # Adaptive confirmation multipliers
    volume_multiplier = pd.Series(1.0, index=df.index)
    
    for i in range(1, len(df)):
        if velocity_direction.iloc[i] == volume_velocity.iloc[i]:
            if volume_change_ratio.iloc[i] > 1.5:
                volume_multiplier.iloc[i] = 1.8  # Strong confirmation
            elif volume_change_ratio.iloc[i] > 1.1:
                volume_multiplier.iloc[i] = 1.3  # Moderate confirmation
            else:
                volume_multiplier.iloc[i] = 1.0  # Weak confirmation
        else:
            volume_multiplier.iloc[i] = 0.4  # Negative confirmation
    
    # Volume trend consistency
    volume_trend_strength = volume_multiplier.rolling(window=3, min_periods=1).mean()
    
    # Core factor calculation
    base_factor = decay_weighted_persistence * velocity_acceleration * velocity_direction
    
    # Apply volume confirmation
    volume_adjusted_factor = base_factor * volume_multiplier * volume_trend_strength
    
    # Volatility normalization
    range_volatility = (df['high'] - df['low']).rolling(window=3, min_periods=1).mean()
    velocity_volatility = velocity.rolling(window=5, min_periods=1).std()
    
    # Combined volatility adjustment
    volatility_adjusted = volume_adjusted_factor / (range_volatility * velocity_volatility.replace(0, 0.0001))
    
    # Adaptive scaling
    def adaptive_scale(x):
        if abs(x) > 1:
            return np.sign(x) * np.log1p(abs(x))
        else:
            return x
    
    alpha = volatility_adjusted.apply(adaptive_scale)
    
    # Fill NaN values
    alpha = alpha.fillna(0)
    
    return alpha
