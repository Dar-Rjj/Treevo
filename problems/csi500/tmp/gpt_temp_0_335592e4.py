import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Momentum Acceleration Component
    # Short-Term Momentum (5-day)
    short_term_momentum = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    
    # Medium-Term Momentum (20-day)
    medium_term_momentum = (data['close'] - data['close'].shift(20)) / data['close'].shift(20)
    
    # Momentum Acceleration
    momentum_acceleration = (short_term_momentum - medium_term_momentum) / (np.abs(medium_term_momentum) + 1e-8)
    
    # Volume Acceleration Component
    # Volume Momentum (5-day)
    volume_momentum = (data['volume'] - data['volume'].shift(5)) / (data['volume'].shift(5) + 1e-8)
    
    # Volume Trend (20-day)
    volume_trend = (data['volume'] - data['volume'].shift(20)) / (data['volume'].shift(20) + 1e-8)
    
    # Volume Acceleration
    volume_acceleration = (volume_momentum - volume_trend) / (np.abs(volume_trend) + 1e-8)
    
    # Regime Detection System
    # Price Regime Classification
    current_range = data['high'] - data['low']
    avg_range = (data['high'] - data['low']).rolling(window=10, min_periods=1).mean()
    
    # Price regime classification
    price_stable = current_range < (0.7 * avg_range)
    price_volatile = current_range > (1.3 * avg_range)
    price_normal = ~(price_stable | price_volatile)
    
    # Volume Regime Classification
    volume_median = data['volume'].rolling(window=10, min_periods=1).median()
    volume_normal = (data['volume'] >= (0.8 * volume_median)) & (data['volume'] <= (1.2 * volume_median))
    volume_extreme = ~volume_normal
    
    # Dynamic Factor Integration
    # Momentum-Volume Alignment
    momentum_volume_alignment = momentum_acceleration * volume_acceleration
    # Apply signed cube root for non-linearity
    aligned_factor = np.sign(momentum_volume_alignment) * np.abs(momentum_volume_alignment) ** (1/3)
    
    # Regime-Based Adjustment
    regime_weight = pd.Series(1.0, index=data.index)
    
    # Price-Stable & Volume-Normal: Apply direct weighting
    stable_normal_mask = price_stable & volume_normal
    regime_weight[stable_normal_mask] = 1.2
    
    # Price-Volatile & Volume-Extreme: Apply inverse weighting
    volatile_extreme_mask = price_volatile & volume_extreme
    regime_weight[volatile_extreme_mask] = 0.8
    
    # Mixed regimes: Apply neutral weighting (already 1.0)
    
    # Apply regime weighting
    regime_adjusted_factor = aligned_factor * regime_weight
    
    # Signal Enhancement
    # Calculate regime persistence (how many consecutive days in current regime)
    price_regime = pd.Series(0, index=data.index)
    price_regime[price_stable] = 1
    price_regime[price_volatile] = 2
    
    volume_regime = pd.Series(0, index=data.index)
    volume_regime[volume_normal] = 1
    volume_regime[volume_extreme] = 2
    
    # Calculate persistence for both regimes
    price_persistence = price_regime.groupby((price_regime != price_regime.shift(1)).cumsum()).cumcount() + 1
    volume_persistence = volume_regime.groupby((volume_regime != volume_regime.shift(1)).cumsum()).cumcount() + 1
    
    # Use regime persistence to adjust smoothing parameter
    # Higher persistence = stronger smoothing (lower alpha)
    persistence_factor = (price_persistence + volume_persistence) / 2
    smoothing_alpha = 2 / (persistence_factor + 2)  # Alpha ranges from 0.5 to 1.0
    
    # Apply exponential smoothing with dynamic alpha
    final_factor = pd.Series(index=data.index, dtype=float)
    final_factor.iloc[0] = regime_adjusted_factor.iloc[0] if not pd.isna(regime_adjusted_factor.iloc[0]) else 0
    
    for i in range(1, len(data)):
        if pd.isna(regime_adjusted_factor.iloc[i]) or pd.isna(final_factor.iloc[i-1]):
            final_factor.iloc[i] = regime_adjusted_factor.iloc[i] if not pd.isna(regime_adjusted_factor.iloc[i]) else 0
        else:
            alpha = smoothing_alpha.iloc[i]
            final_factor.iloc[i] = alpha * regime_adjusted_factor.iloc[i] + (1 - alpha) * final_factor.iloc[i-1]
    
    return final_factor
