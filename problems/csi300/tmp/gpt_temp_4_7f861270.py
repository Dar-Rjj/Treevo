import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(data):
    """
    Momentum Divergence with Volume Confirmation factor
    Captures acceleration/deceleration patterns confirmed by volume trends
    """
    close = data['close']
    volume = data['volume']
    
    # Calculate Short-Term Momentum (5-day)
    short_term_momentum = close / close.shift(5) - 1
    
    # Calculate Medium-Term Momentum (20-day)
    medium_term_momentum = close / close.shift(20) - 1
    
    # Calculate Momentum Divergence
    momentum_divergence = short_term_momentum - medium_term_momentum
    
    # Calculate Volume Trend Slope (5-day window)
    volume_trend_slope = volume.rolling(window=6).apply(
        lambda x: linregress(range(len(x)), x)[0] if len(x) == 6 else np.nan,
        raw=False
    )
    
    # Normalize volume trend slope by recent volume levels
    volume_normalized = volume_trend_slope / volume.rolling(window=5).mean()
    
    # Apply Volume Confirmation
    factor = momentum_divergence * volume_normalized
    
    return factor
