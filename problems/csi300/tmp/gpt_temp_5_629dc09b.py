import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Dimensional Price Memory Momentum Alpha Factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Calculate required periods
    for i in range(max(8, len(data))):
        if i < 8:
            factor.iloc[i] = 0
            continue
            
        current_idx = data.index[i]
        
        # Fractal Price Memory Framework
        # Short-Term Memory (3-day)
        short_price_memory = (data['close'].iloc[i] - data['close'].iloc[i-3]) / (
            max(data['high'].iloc[i-2:i+1]) - min(data['low'].iloc[i-2:i+1]) + 1e-8
        )
        
        short_persistence = sum(
            np.sign(data['close'].iloc[j] - data['close'].iloc[j-1]) 
            for j in range(i-2, i+1)
        )
        
        short_decay_denom = data['close'].iloc[i-1] - data['close'].iloc[i-3]
        short_decay = (data['close'].iloc[i] - data['close'].iloc[i-1]) / (short_decay_denom + 1e-8 if short_decay_denom != 0 else 1e-8)
        
        # Medium-Term Memory (8-day)
        medium_price_memory = (data['close'].iloc[i] - data['close'].iloc[i-8]) / (
            max(data['high'].iloc[i-7:i+1]) - min(data['low'].iloc[i-7:i+1]) + 1e-8
        )
        
        medium_persistence = sum(
            np.sign(data['close'].iloc[j] - data['close'].iloc[j-1]) 
            for j in range(i-7, i+1)
        )
        
        medium_decay_denom = data['close'].iloc[i-4] - data['close'].iloc[i-8]
        medium_decay = (data['close'].iloc[i] - data['close'].iloc[i-4]) / (medium_decay_denom + 1e-8 if medium_decay_denom != 0 else 1e-8)
        
        # Memory Scale Interaction
        memory_convergence = short_price_memory / (medium_price_memory + 1e-8)
        memory_divergence = short_persistence - medium_persistence
        memory_acceleration = short_decay / (medium_decay + 1e-8)
        
        # Price Memory Asymmetry
        up_memory_strength = 0
        down_memory_strength = 0
        for j in range(i-4, i+1):
            if data['close'].iloc[j] > data['close'].iloc[j-1]:
                up_memory_strength += data['close'].iloc[j] - data['close'].iloc[j-1]
            elif data['close'].iloc[j] < data['close'].iloc[j-1]:
                down_memory_strength += data['close'].iloc[j-1] - data['close'].iloc[j]
        
        memory_asymmetry_ratio = up_memory_strength / (down_memory_strength + 1e-8)
        
        # Memory Regime Detection
        memory_regime = 1.0  # Default transition regime
        if short_persistence > 2:
            memory_regime = 1.5  # Strong memory regime
        elif short_persistence < -2:
            memory_regime = 0.5  # Weak memory regime
        
        # Volume-Memory Integration
        volume_avg = np.mean([data['volume'].iloc[j] for j in range(i-4, i)])
        volume_memory_ratio = data['volume'].iloc[i] / (volume_avg + 1e-8)
        
        # Calculate volume-price correlation over 5 days
        if i >= 12:
            price_memory_ratios = []
            volume_ratios = []
            for j in range(i-4, i+1):
                short_pr = (data['close'].iloc[j] - data['close'].iloc[j-3]) / (
                    max(data['high'].iloc[j-2:j+1]) - min(data['low'].iloc[j-2:j+1]) + 1e-8
                )
                vol_avg = np.mean([data['volume'].iloc[k] for k in range(j-4, j)])
                vol_ratio = data['volume'].iloc[j] / (vol_avg + 1e-8)
                price_memory_ratios.append(short_pr)
                volume_ratios.append(vol_ratio)
            
            volume_memory_correlation = np.corrcoef(price_memory_ratios, volume_ratios)[0,1]
            if np.isnan(volume_memory_correlation):
                volume_memory_correlation = 0
        else:
            volume_memory_correlation = 0
        
        # Asymmetric Volume Memory
        current_price_ratio = short_price_memory
        up_confirming_volume = data['volume'].iloc[i] if (data['close'].iloc[i] > data['close'].iloc[i-1] and current_price_ratio > 0) else 0
        down_confirming_volume = data['volume'].iloc[i] if (data['close'].iloc[i] < data['close'].iloc[i-1] and current_price_ratio < 0) else 0
        
        up_contradictory_volume = data['volume'].iloc[i] if (data['close'].iloc[i] > data['close'].iloc[i-1] and current_price_ratio < 0) else 0
        down_contradictory_volume = data['volume'].iloc[i] if (data['close'].iloc[i] < data['close'].iloc[i-1] and current_price_ratio > 0) else 0
        
        confirmation_ratio = up_confirming_volume / (down_confirming_volume + 1e-8)
        contradiction_ratio = up_contradictory_volume / (down_contradictory_volume + 1e-8)
        
        # Volume Confirmation Multiplier
        volume_confirmation = 1.0
        if up_confirming_volume > 2 * down_confirming_volume and up_confirming_volume > 0:
            volume_confirmation = 1.2
        elif down_confirming_volume > 2 * up_confirming_volume and down_confirming_volume > 0:
            volume_confirmation = 1.2
        elif (up_contradictory_volume > 0 or down_contradictory_volume > 0) and confirmation_ratio < 0.5:
            volume_confirmation = 0.5
        elif volume_memory_ratio < 0.8:
            volume_confirmation = 0.8
        
        # Price-Level Memory Anchoring
        recent_high = max(data['high'].iloc[i-4:i+1])
        recent_low = min(data['low'].iloc[i-4:i+1])
        
        recent_high_memory = data['close'].iloc[i] / (recent_high + 1e-8)
        recent_low_memory = data['close'].iloc[i] / (recent_low + 1e-8)
        memory_level_ratio = recent_high_memory / (recent_low_memory + 1e-8)
        
        # Memory Breakout Detection
        prev_high = max(data['high'].iloc[i-4:i])
        prev_low = min(data['low'].iloc[i-4:i])
        
        high_breakout = data['close'].iloc[i] > prev_high
        low_breakout = data['close'].iloc[i] < prev_low
        
        breakout_strength = (data['close'].iloc[i] - prev_high) / (prev_high - prev_low + 1e-8) if high_breakout else 0
        if low_breakout:
            breakout_strength = (data['close'].iloc[i] - prev_low) / (prev_high - prev_low + 1e-8)
        
        # Breakout Multiplier
        breakout_multiplier = 1.0
        if high_breakout and breakout_strength > 0.1:
            breakout_multiplier = 2.0
        elif low_breakout and breakout_strength < -0.1:
            breakout_multiplier = -2.0
        
        # Memory Support/Resistance Framework
        memory_support = min(data['low'].iloc[i-4:i+1])
        memory_resistance = max(data['high'].iloc[i-4:i+1])
        memory_level_proximity = (data['close'].iloc[i] - memory_support) / (memory_resistance - memory_support + 1e-8)
        
        # Multi-Timeframe Memory Momentum
        short_memory_momentum = short_price_memory * short_persistence
        medium_memory_momentum = medium_price_memory * medium_persistence
        memory_momentum_ratio = short_memory_momentum / (medium_memory_momentum + 1e-8)
        
        # Composite Memory Alpha Factor
        base_memory = memory_momentum_ratio * memory_asymmetry_ratio
        volume_enhanced = base_memory * (1 + volume_memory_correlation)
        level_anchored = volume_enhanced * memory_level_proximity
        
        # Apply all multipliers
        final_alpha = level_anchored * memory_regime * breakout_multiplier * volume_confirmation
        
        # Normalize to desired range and avoid extreme values
        final_alpha = np.clip(final_alpha, -2.0, 2.0)
        
        factor.iloc[i] = final_alpha
    
    return factor
