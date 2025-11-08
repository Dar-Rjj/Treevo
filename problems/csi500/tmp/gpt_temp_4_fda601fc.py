import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Dynamic Volume-Momentum Alignment with Volatility-Weighted Persistence
    
    This alpha factor combines momentum persistence, volume-momentum alignment,
    and volatility context to generate predictive signals.
    """
    
    # Initialize result series
    result = pd.Series(index=data.index, dtype=float)
    
    # Small epsilon to avoid division by zero
    eps = 1e-8
    
    # Pre-calculate basic components
    close = data['close']
    open_price = data['open']
    high = data['high']
    low = data['low']
    volume = data['volume']
    
    # 1. Core Momentum Signal Generation
    # Intraday Momentum Strength
    daily_range = high - low + eps
    raw_momentum = (close - open_price) / daily_range
    momentum_direction = np.sign(close - open_price)
    
    # Momentum Persistence Framework
    persistence_score = pd.Series(0.0, index=data.index)
    consecutive_days = pd.Series(0, index=data.index)
    lambda_decay = 0.95
    
    for i in range(1, len(data)):
        current_dir = momentum_direction.iloc[i]
        prev_dir = momentum_direction.iloc[i-1]
        
        # Dynamic Persistence Tracking
        if current_dir == prev_dir and current_dir != 0:
            consecutive_days.iloc[i] = consecutive_days.iloc[i-1] + 1
        else:
            consecutive_days.iloc[i] = 1 if current_dir != 0 else 0
        
        # Persistence strength multiplier
        persistence_strength = min(consecutive_days.iloc[i] * 0.2, 2.0)  # Cap at 2.0
        
        # Time-Decayed Persistence Score
        if i == 1:
            persistence_score.iloc[i] = persistence_strength
        else:
            persistence_score.iloc[i] = (lambda_decay * persistence_score.iloc[i-1] + 
                                       persistence_strength)
    
    # 2. Volume-Momentum Alignment System
    volume_change_ratio = volume / volume.shift(1)
    volume_momentum_direction = np.sign(volume - volume.shift(1))
    
    # Alignment Strength Assessment
    alignment_strength = pd.Series(0.0, index=data.index)
    alignment_direction = pd.Series(0, index=data.index)
    
    for i in range(1, len(data)):
        price_dir = momentum_direction.iloc[i]
        volume_dir = volume_momentum_direction.iloc[i]
        
        # Direction Alignment Score
        if price_dir == volume_dir and price_dir != 0:
            alignment_direction.iloc[i] = 1  # Positive alignment
        elif price_dir != volume_dir and price_dir != 0 and volume_dir != 0:
            alignment_direction.iloc[i] = -1  # Negative alignment
        else:
            alignment_direction.iloc[i] = 0  # Neutral alignment
        
        # Magnitude-Based Alignment Strength
        if not pd.isna(volume_change_ratio.iloc[i]):
            alignment_strength.iloc[i] = abs(volume_change_ratio.iloc[i] - 1)
    
    # 3. Volatility Context Integration
    intraday_volatility = high - low
    recent_volatility = pd.Series(0.0, index=data.index)
    
    # 5-Day Volatility Baseline
    for i in range(4, len(data)):
        recent_volatility.iloc[i] = np.mean([high.iloc[i-j] - low.iloc[i-j] for j in range(5)])
    
    # 4. Final Alpha Factor Construction
    for i in range(4, len(data)):
        # Combined Signal Generation
        base_signal = persistence_score.iloc[i] * alignment_strength.iloc[i]
        
        # Apply directional alignment and volatility scaling
        if intraday_volatility.iloc[i] > eps:
            volatility_factor = intraday_volatility.iloc[i] / (recent_volatility.iloc[i] + eps)
            factor_value = (base_signal * alignment_direction.iloc[i] * volatility_factor) / intraday_volatility.iloc[i]
        else:
            factor_value = 0.0
        
        result.iloc[i] = factor_value
    
    # Fill initial NaN values with 0
    result = result.fillna(0)
    
    return result
