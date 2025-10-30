import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    """
    Price-Volume Convergence Divergence Factor
    Combines short-term and medium-term price and volume trends to detect
    convergence/divergence patterns that may predict future returns.
    """
    # Extract price and volume data
    close = df['close']
    volume = df['volume']
    
    # Initialize result series
    factor = pd.Series(index=df.index, dtype=float)
    
    # Calculate trends using linear regression
    for i in range(len(df)):
        if i < 6:  # Need at least 6 days of data
            factor.iloc[i] = 0
            continue
            
        # Short-term price trend (3-day: t-2 to t)
        price_short_window = close.iloc[i-2:i+1]
        price_short_slope = stats.linregress(range(len(price_short_window)), price_short_window.values)[0]
        
        # Medium-term price trend (6-day: t-5 to t)
        price_medium_window = close.iloc[i-5:i+1]
        price_medium_slope = stats.linregress(range(len(price_medium_window)), price_medium_window.values)[0]
        
        # Short-term volume trend (3-day: t-2 to t)
        volume_short_window = volume.iloc[i-2:i+1]
        volume_short_slope = stats.linregress(range(len(volume_short_window)), volume_short_window.values)[0]
        
        # Medium-term volume trend (6-day: t-5 to t)
        volume_medium_window = volume.iloc[i-5:i+1]
        volume_medium_slope = stats.linregress(range(len(volume_medium_window)), volume_medium_window.values)[0]
        
        # Price trend alignment
        if abs(price_medium_slope) > 1e-8:  # Avoid division by zero
            price_alignment = price_short_slope / price_medium_slope
        else:
            price_alignment = 0
        
        # Volume trend alignment
        if abs(volume_medium_slope) > 1e-8:
            volume_alignment = volume_short_slope / volume_medium_slope
        else:
            volume_alignment = 0
        
        # Price-volume synchronization
        price_trend_direction = 1 if price_short_slope > 0 else -1
        volume_trend_direction = 1 if volume_short_slope > 0 else -1
        synchronization = price_trend_direction * volume_trend_direction
        
        # Trend strength indicator
        price_trend_strength = abs(price_short_slope) + abs(price_medium_slope)
        volume_trend_strength = abs(volume_short_slope) + abs(volume_medium_slope)
        trend_strength = price_trend_strength * (1 + volume_trend_strength)
        
        # Convergence score
        alignment_score = price_alignment * volume_alignment
        convergence_score = alignment_score * synchronization
        
        # Composite signal
        signal = trend_strength * convergence_score
        
        # Apply direction based on convergence/divergence
        if price_alignment > 1 and volume_alignment > 1 and synchronization > 0:
            # Bullish convergence: both trends accelerating with aligned direction
            factor.iloc[i] = abs(signal)
        elif price_alignment < 1 and volume_alignment < 1 and synchronization > 0:
            # Bearish convergence: both trends decelerating with aligned direction
            factor.iloc[i] = -abs(signal)
        elif synchronization < 0:
            # Divergence: price and volume moving in opposite directions
            factor.iloc[i] = -signal * 0.5  # Weaker signal for divergence
        else:
            # Neutral or unclear pattern
            factor.iloc[i] = signal * 0.25
    
    return factor
