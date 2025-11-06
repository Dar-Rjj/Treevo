import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price-Volume Momentum Divergence Factor
    Combines short-term vs medium-term momentum in both price and volume
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate price slopes using linear regression
    def calc_slope(series, window):
        x = np.arange(window)
        slopes = []
        for i in range(len(series)):
            if i < window - 1:
                slopes.append(np.nan)
            else:
                y = series.iloc[i-window+1:i+1].values
                if len(y) == window:
                    slope = (window * np.sum(x * y) - np.sum(x) * np.sum(y)) / (window * np.sum(x**2) - np.sum(x)**2)
                    slopes.append(slope)
                else:
                    slopes.append(np.nan)
        return pd.Series(slopes, index=series.index)
    
    # Price momentum divergence
    price_slope_short = calc_slope(data['close'], 5)
    price_slope_medium = calc_slope(data['close'], 20)
    price_ratio = price_slope_short / price_slope_medium
    
    # Volume momentum divergence  
    volume_slope_short = calc_slope(data['volume'], 5)
    volume_slope_medium = calc_slope(data['volume'], 20)
    volume_ratio = volume_slope_short / volume_slope_medium
    
    # Final factor: combine price and volume divergence with direction
    factor = price_ratio * volume_ratio
    factor = np.sign(factor) * np.sqrt(np.abs(factor))
    
    return factor
