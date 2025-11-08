import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Range-Normalized Momentum Calculation
    # Compute Intraday Price Movement
    price_change = data['close'] - data['open']
    daily_range = data['high'] - data['low']
    
    # Calculate Normalized Momentum with epsilon to handle zero range
    epsilon = 1e-8
    momentum = price_change / (daily_range + epsilon)
    
    # Momentum Persistence Analysis
    # Track Direction Persistence
    direction = np.sign(momentum)
    
    # Initialize persistence tracking
    consecutive_days = pd.Series(0, index=data.index)
    persistence_score = pd.Series(0.0, index=data.index)
    
    # Calculate consecutive same-direction days and persistence score
    for i in range(1, len(data)):
        if direction.iloc[i] == direction.iloc[i-1] and direction.iloc[i] != 0:
            consecutive_days.iloc[i] = consecutive_days.iloc[i-1] + 1
        else:
            consecutive_days.iloc[i] = 1 if direction.iloc[i] != 0 else 0
        
        if consecutive_days.iloc[i] > 0:
            base_persistence = consecutive_days.iloc[i] * abs(momentum.iloc[i])
            decayed_persistence = base_persistence * (0.95 ** (consecutive_days.iloc[i] - 1))
            persistence_score.iloc[i] = direction.iloc[i] * decayed_persistence
    
    # Volume Momentum Alignment
    # Compute Volume Momentum
    volume_momentum = data['volume'] / data['volume'].shift(1) - 1
    
    # Volume-Price Alignment Analysis
    alignment_score = pd.Series(0.0, index=data.index)
    strength_multiplier = pd.Series(0.0, index=data.index)
    
    for i in range(1, len(data)):
        if not pd.isna(volume_momentum.iloc[i]) and direction.iloc[i] != 0:
            # Check direction alignment
            if np.sign(volume_momentum.iloc[i]) == direction.iloc[i]:
                alignment_score.iloc[i] = 1.0
            else:
                alignment_score.iloc[i] = -1.0
            
            # Strength multiplier
            strength_multiplier.iloc[i] = abs(volume_momentum.iloc[i])
    
    # Combine Momentum and Volume Signals
    core_signal = persistence_score * alignment_score * strength_multiplier
    
    # Apply Volatility Context
    # Calculate recent volatility proxy (5-day average of daily ranges)
    recent_volatility = daily_range.rolling(window=5, min_periods=1).mean()
    
    # Volatility adjustment
    final_factor = core_signal / (recent_volatility + epsilon)
    
    return final_factor
