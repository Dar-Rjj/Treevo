import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Momentum with Volume Persistence alpha factor
    """
    # Price data
    close = df['close']
    high = df['high']
    low = df['low']
    open_price = df['open']
    volume = df['volume']
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate all required components
    for i in range(len(df)):
        if i < 10:  # Need at least 10 days of data
            result.iloc[i] = 0
            continue
            
        # Price Momentum Components
        momentum_1d = close.iloc[i] - close.iloc[i-1]
        momentum_3d = close.iloc[i] - close.iloc[i-2]
        momentum_10d = close.iloc[i] - close.iloc[i-9]
        
        opening_gap = open_price.iloc[i] - close.iloc[i-1]
        daily_range = high.iloc[i] - low.iloc[i]
        close_strength = close.iloc[i] - (high.iloc[i] + low.iloc[i]) / 2
        
        # Volatility Framework
        # Short-term volatility (3-day average range)
        short_term_vol = (
            (high.iloc[i] - low.iloc[i]) + 
            (high.iloc[i-1] - low.iloc[i-1]) + 
            (high.iloc[i-2] - low.iloc[i-2])
        ) / 3
        
        # Medium-term volatility (10-day average range)
        medium_term_vol = sum(high.iloc[i-j] - low.iloc[i-j] for j in range(10)) / 10
        
        volatility_ratio = short_term_vol / medium_term_vol if medium_term_vol != 0 else 1.0
        
        # Volatility-scaled momentum
        vsm_1d = momentum_1d / daily_range if daily_range != 0 else 0
        vsm_3d = momentum_3d / sum(high.iloc[i-j] - low.iloc[i-j] for j in range(3)) if sum(high.iloc[i-j] - low.iloc[i-j] for j in range(3)) != 0 else 0
        vsm_10d = momentum_10d / sum(high.iloc[i-j] - low.iloc[i-j] for j in range(10)) if sum(high.iloc[i-j] - low.iloc[i-j] for j in range(10)) != 0 else 0
        
        # Volatility regime
        if volatility_ratio > 1.15:
            vol_regime_multiplier = 0.6
        elif volatility_ratio >= 0.85:
            vol_regime_multiplier = 1.0
        else:
            vol_regime_multiplier = 1.4
        
        # Volume Persistence
        volume_change = volume.iloc[i] - volume.iloc[i-1]
        volume_direction = np.sign(volume_change)
        
        # Calculate volume streak (consecutive same direction days)
        volume_streak = 1
        for j in range(1, min(10, i)):
            if np.sign(volume.iloc[i-j] - volume.iloc[i-j-1]) == volume_direction:
                volume_streak += 1
            else:
                break
        
        # Volume-momentum alignment
        direction_match = 1 if np.sign(momentum_1d) == np.sign(volume_change) else 0
        
        # Calculate alignment streak
        alignment_streak = 1 if direction_match else 0
        for j in range(1, min(5, i)):
            prev_momentum = close.iloc[i-j] - close.iloc[i-j-1]
            prev_volume_change = volume.iloc[i-j] - volume.iloc[i-j-1]
            if np.sign(prev_momentum) == np.sign(prev_volume_change):
                alignment_streak += 1
            else:
                break
        
        alignment_strength = alignment_streak * abs(momentum_1d)
        
        # Volume regime
        volume_ratio = sum(volume.iloc[i-j] for j in range(3)) / sum(volume.iloc[i-j] for j in range(10)) if sum(volume.iloc[i-j] for j in range(10)) != 0 else 1.0
        
        if volume_ratio > 1.1:
            volume_regime_multiplier = 1.3
        elif volume_ratio >= 0.9:
            volume_regime_multiplier = 1.0
        else:
            volume_regime_multiplier = 0.7
        
        # Adaptive Factor Construction
        # Base momentum signal
        multi_timeframe_blend = (4 * vsm_1d + 3 * vsm_3d + vsm_10d) / 8
        base_signal = multi_timeframe_blend * np.log(volume.iloc[i] + 1)
        persistence_boosted = base_signal * (1 + volume_streak / 10)
        
        # Momentum acceleration
        acceleration = vsm_3d - vsm_10d
        acceleration_direction = np.sign(acceleration)
        acceleration_confirmation = 1 + 0.2 * acceleration_direction
        
        # Final composite score
        raw_factor = persistence_boosted
        regime_adjusted = raw_factor * vol_regime_multiplier * volume_regime_multiplier
        final_factor = regime_adjusted * acceleration_confirmation
        
        result.iloc[i] = final_factor
    
    return result
