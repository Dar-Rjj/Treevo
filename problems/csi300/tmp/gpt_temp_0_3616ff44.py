import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    """
    Trend Acceleration with Volume Confirmation factor
    Combines price trend acceleration with volume trend confirmation
    """
    close = df['close']
    volume = df['volume']
    
    def rolling_slope(series, window):
        """Calculate rolling linear regression slope"""
        slopes = pd.Series(index=series.index, dtype=float)
        for i in range(window - 1, len(series)):
            if i >= window - 1:
                y_values = series.iloc[i - window + 1:i + 1].values
                if len(np.unique(y_values)) > 1:  # Ensure we have variation
                    x_values = np.arange(len(y_values))
                    slope, _, _, _, _ = linregress(x_values, y_values)
                    slopes.iloc[i] = slope
                else:
                    slopes.iloc[i] = 0
        return slopes
    
    # Calculate price trends
    short_trend = rolling_slope(close, 5)
    medium_trend = rolling_slope(close, 20)
    
    # Calculate price acceleration (difference between trends)
    price_acceleration = short_trend - medium_trend
    
    # Calculate volume trend
    volume_trend = rolling_slope(volume, 5)
    
    # Apply volume confirmation
    # Use sign of volume trend to confirm or contradict price acceleration
    volume_confirmation = np.sign(volume_trend)
    
    # Final factor: price acceleration multiplied by volume trend sign
    factor = price_acceleration * volume_confirmation
    
    return factor
