import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Momentum Divergence with Volume Spike alpha factor
    Combines price momentum acceleration with volume confirmation signals
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Recent Price Momentum calculations
    # Price Rate of Change (5-day)
    price_roc = data['close'].pct_change(periods=5)
    
    # Momentum Acceleration (rate of change of momentum)
    momentum_accel = price_roc.pct_change(periods=4)
    
    # Volume Spike Detection
    # 20-day average volume
    avg_volume_20 = data['volume'].rolling(window=20).mean()
    
    # Volume ratio (current volume / 20-day average)
    volume_ratio = data['volume'] / avg_volume_20
    
    # Volume spike flag (ratio > 2.0)
    volume_spike = (volume_ratio > 2.0).astype(int)
    
    # Momentum-Volume Divergence scoring
    alpha_scores = pd.Series(index=data.index, dtype=float)
    
    for i in range(len(data)):
        if i < 20:  # Need enough data for volume average
            alpha_scores.iloc[i] = 0
            continue
            
        current_momentum = price_roc.iloc[i]
        current_accel = momentum_accel.iloc[i]
        current_volume_spike = volume_spike.iloc[i]
        
        # Base momentum score (normalized)
        momentum_score = np.tanh(current_momentum * 10)  # Scale and bound
        
        # Volume confirmation multiplier
        if current_volume_spike:
            if current_momentum > 0 and current_accel > 0:
                # Strong continuation: positive momentum with volume spike
                volume_multiplier = 2.0
            elif current_momentum < 0 and current_accel < 0:
                # Strong reversal confirmation: negative momentum with volume spike
                volume_multiplier = -1.5
            else:
                # Mixed signals with volume spike
                volume_multiplier = 1.2
        else:
            if current_momentum > 0:
                # Weak positive momentum without volume support
                volume_multiplier = 0.6
            else:
                # Weak negative momentum without volume support
                volume_multiplier = -0.4
        
        # Acceleration adjustment
        accel_adjustment = 1.0 + (current_accel * 5)  # Scale acceleration effect
        
        # Final alpha score
        alpha_score = momentum_score * volume_multiplier * accel_adjustment
        
        # Apply smoothing to reduce noise
        if i > 0:
            alpha_scores.iloc[i] = 0.7 * alpha_score + 0.3 * alpha_scores.iloc[i-1]
        else:
            alpha_scores.iloc[i] = alpha_score
    
    return alpha_scores
