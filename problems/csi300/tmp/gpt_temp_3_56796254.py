import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(data):
    """
    Momentum-Volume Divergence Factor
    Combines momentum divergence between short-term and medium-term periods
    with volume trend confirmation for enhanced predictive power.
    """
    df = data.copy()
    close = df['close']
    volume = df['volume']
    
    # Calculate Short-term Momentum
    mom_3d = close / close.shift(3) - 1
    mom_5d = close / close.shift(5) - 1
    short_term_mom = (mom_3d + mom_5d) / 2
    
    # Calculate Medium-term Momentum
    mom_10d = close / close.shift(10) - 1
    mom_15d = close / close.shift(15) - 1
    medium_term_mom = (mom_10d + mom_15d) / 2
    
    # Calculate Volume Trends using linear regression slopes
    def calc_volume_slope(series, window):
        slopes = pd.Series(index=series.index, dtype=float)
        for i in range(window-1, len(series)):
            if i >= window-1:
                y = series.iloc[i-window+1:i+1].values
                x = np.arange(len(y))
                slope, _, _, _, _ = linregress(x, y)
                slopes.iloc[i] = slope
        return slopes
    
    vol_slope_5d = calc_volume_slope(volume, 5)
    vol_slope_10d = calc_volume_slope(volume, 10)
    volume_trend = (vol_slope_5d + vol_slope_10d) / 2
    
    # Compute Momentum Divergence
    momentum_divergence = short_term_mom - medium_term_mom
    divergence_direction = np.sign(momentum_divergence)
    
    # Combine with Volume Confirmation
    def get_volume_confirmation(divergence, volume_trend):
        volume_confirmation = pd.Series(index=divergence.index, dtype=float)
        
        # Strong positive: momentum divergence up + volume increasing
        strong_pos_mask = (divergence > 0) & (volume_trend > 0)
        volume_confirmation[strong_pos_mask] = 1.5
        
        # Moderate positive: momentum divergence up + volume stable
        moderate_pos_mask = (divergence > 0) & (volume_trend <= 0)
        volume_confirmation[moderate_pos_mask] = 1.0
        
        # Negative: momentum divergence down + volume increasing
        negative_mask = (divergence < 0) & (volume_trend > 0)
        volume_confirmation[negative_mask] = -1.2
        
        # Weak negative: momentum divergence down + volume decreasing
        weak_neg_mask = (divergence < 0) & (volume_trend <= 0)
        volume_confirmation[weak_neg_mask] = -0.8
        
        return volume_confirmation
    
    volume_confirmation_weights = get_volume_confirmation(divergence_direction, volume_trend)
    
    # Final factor = momentum divergence × volume confirmation weight
    factor = momentum_divergence * volume_confirmation_weights
    
    return factor
