import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Timeframe Momentum Calculation
    data['momentum_2d'] = data['close'] / data['close'].shift(2) - 1
    data['momentum_5d'] = data['close'] / data['close'].shift(5) - 1
    data['momentum_10d'] = data['close'] / data['close'].shift(10) - 1
    data['momentum_20d'] = data['close'] / data['close'].shift(20) - 1
    
    # Momentum Convergence Assessment
    # Count aligned directions across timeframes
    momentum_signs = pd.DataFrame({
        'm2_sign': np.sign(data['momentum_2d']),
        'm5_sign': np.sign(data['momentum_5d']),
        'm10_sign': np.sign(data['momentum_10d']),
        'm20_sign': np.sign(data['momentum_20d'])
    })
    
    # Count how many timeframes have the same direction as the majority
    majority_sign = momentum_signs.mode(axis=1).iloc[:, 0]
    alignment_count = (momentum_signs.T == majority_sign).T.sum(axis=1)
    
    # Average aligned momentum values (only consider those aligned with majority)
    aligned_momentum = []
    for i in range(len(data)):
        row_signs = momentum_signs.iloc[i]
        row_momentums = [data['momentum_2d'].iloc[i], data['momentum_5d'].iloc[i], 
                        data['momentum_10d'].iloc[i], data['momentum_20d'].iloc[i]]
        
        aligned_values = [m for j, m in enumerate(row_momentums) 
                         if row_signs.iloc[j] == majority_sign.iloc[i]]
        aligned_momentum.append(np.mean(aligned_values) if aligned_values else 0)
    
    data['aligned_momentum_avg'] = aligned_momentum
    data['momentum_alignment'] = alignment_count / 4.0  # Normalize to 0-1
    
    # Volume-Weighted Adjustment
    data['volume_momentum'] = data['volume'] / data['volume'].shift(5) - 1
    data['volume_multiplier'] = np.sign(data['volume_momentum']) * np.abs(data['volume_momentum'])
    
    # Price-Volume Divergence Detection
    momentum_direction = np.sign(data['aligned_momentum_avg'])
    volume_direction = np.sign(data['volume_momentum'])
    
    # Divergence occurs when price and volume move in opposite directions
    divergence_signal = momentum_direction * volume_direction
    data['divergence_modifier'] = np.where(divergence_signal < 0, -1, 1)
    
    # Composite Alpha Generation
    # Combine convergence strength with volume-weighted adjustment
    convergence_strength = data['momentum_alignment'] * data['aligned_momentum_avg']
    volume_adjusted = convergence_strength * data['volume_multiplier']
    
    # Apply divergence modifier
    alpha_factor = volume_adjusted * data['divergence_modifier']
    
    return alpha_factor
