import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    """
    Price-Volume Divergence Momentum factor that compares price and volume trend directions
    across short-term (5-day) and medium-term (20-day) horizons.
    """
    def calculate_slope(series, window):
        """Calculate linear regression slope for given window"""
        slopes = pd.Series(index=series.index, dtype=float)
        for i in range(window-1, len(series)):
            if i >= window-1:
                y_values = series.iloc[i-window+1:i+1].values
                if len(y_values) == window and not np.all(y_values == y_values[0]):
                    x_values = np.arange(window)
                    slope, _, _, _, _ = linregress(x_values, y_values)
                    slopes.iloc[i] = slope
                else:
                    slopes.iloc[i] = 0
        return slopes
    
    # Calculate price trends
    price_short_slope = calculate_slope(df['close'], 5)
    price_medium_slope = calculate_slope(df['close'], 20)
    
    # Calculate volume trends  
    volume_short_slope = calculate_slope(df['volume'], 5)
    volume_medium_slope = calculate_slope(df['volume'], 20)
    
    # Calculate divergence scores
    short_term_divergence = price_short_slope * volume_short_slope
    medium_term_divergence = price_medium_slope * volume_medium_slope
    
    # Apply sign function and scale by magnitude difference
    short_term_score = np.sign(short_term_divergence) * (abs(price_short_slope) - abs(volume_short_slope))
    medium_term_score = np.sign(medium_term_divergence) * (abs(price_medium_slope) - abs(volume_medium_slope))
    
    # Combine short and medium term scores
    final_factor = 0.6 * short_term_score + 0.4 * medium_term_score
    
    return final_factor
