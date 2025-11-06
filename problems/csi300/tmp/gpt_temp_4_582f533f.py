import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price Fractal Efficiency
    # Cumulative price movement over 10 days
    price_movement = (df['close'] - df['close'].shift(10)).abs()
    
    # Straight-line distance between day t-10 and day t
    straight_distance = (df['close'] - df['close'].shift(10)).abs()
    
    # Efficiency ratio (distance / movement)
    # To avoid division by zero, add small epsilon
    price_efficiency = straight_distance / (price_movement + 1e-8)
    price_efficiency = price_efficiency.replace([np.inf, -np.inf], 0)
    
    # Volume Efficiency Pattern
    # Calculate daily returns
    returns = df['close'].pct_change()
    
    # Net volume direction (signed volume × return)
    net_volume_direction = df['volume'] * returns
    
    # Cumulative volume flow over 10 days
    cumulative_volume_flow = net_volume_direction.rolling(window=10, min_periods=1).sum()
    
    # Volume efficiency ratio (net direction / cumulative flow)
    # Use absolute value of cumulative flow to maintain direction
    volume_efficiency = net_volume_direction / (cumulative_volume_flow.abs() + 1e-8)
    volume_efficiency = volume_efficiency.replace([np.inf, -np.inf], 0)
    
    # Regime Transition Detection
    # Efficiency momentum (change in price efficiency)
    efficiency_momentum = price_efficiency.diff(3)
    
    # Volume efficiency divergence detection
    volume_divergence = volume_efficiency - volume_efficiency.rolling(window=5, min_periods=1).mean()
    
    # Identify regime shift periods using rolling z-score
    regime_shift = efficiency_momentum.rolling(window=10, min_periods=1).apply(
        lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-8) if x.std() > 0 else 0
    )
    
    # Transition-Enhanced Factor
    # Combine efficiency metrics with regime weights
    base_factor = price_efficiency * 0.6 + volume_efficiency * 0.4
    
    # Apply transition period enhancement using regime shift detection
    transition_enhancement = 1 + np.tanh(regime_shift.abs()) * np.sign(regime_shift)
    
    # Generate dynamic efficiency prediction signal
    dynamic_factor = base_factor * transition_enhancement
    
    # Final normalization using rolling z-score
    final_factor = dynamic_factor.rolling(window=20, min_periods=1).apply(
        lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-8) if x.std() > 0 else 0
    )
    
    return final_factor
