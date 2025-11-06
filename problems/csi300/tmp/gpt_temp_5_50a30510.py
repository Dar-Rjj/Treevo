import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    """
    Price and Volume Divergence Momentum factor
    Combines short and medium-term price and volume momentum with divergence detection
    """
    # Make a copy to avoid modifying original dataframe
    data = df.copy()
    
    # Initialize result series
    factor_values = pd.Series(index=data.index, dtype=float)
    
    # Calculate price momentum components
    short_price_slope = pd.Series(index=data.index, dtype=float)
    medium_price_slope = pd.Series(index=data.index, dtype=float)
    
    # Calculate volume momentum components  
    short_volume_slope = pd.Series(index=data.index, dtype=float)
    medium_volume_slope = pd.Series(index=data.index, dtype=float)
    
    # Calculate linear regression slopes for price and volume
    for i in range(len(data)):
        if i >= 20:  # Need at least 20 periods for medium-term calculation
            # Short-term price trend (5 periods)
            if i >= 5:
                x_short = np.arange(5)
                y_price_short = data['close'].iloc[i-4:i+1].values
                slope_price_short, _, _, _, _ = stats.linregress(x_short, y_price_short)
                short_price_slope.iloc[i] = slope_price_short
            
            # Medium-term price trend (20 periods)
            x_medium = np.arange(20)
            y_price_medium = data['close'].iloc[i-19:i+1].values
            slope_price_medium, _, _, _, _ = stats.linregress(x_medium, y_price_medium)
            medium_price_slope.iloc[i] = slope_price_medium
            
            # Short-term volume trend (5 periods)
            if i >= 5:
                y_volume_short = data['volume'].iloc[i-4:i+1].values
                slope_volume_short, _, _, _, _ = stats.linregress(x_short, y_volume_short)
                short_volume_slope.iloc[i] = slope_volume_short
            
            # Medium-term volume trend (20 periods)
            y_volume_medium = data['volume'].iloc[i-19:i+1].values
            slope_volume_medium, _, _, _, _ = stats.linregress(x_medium, y_volume_medium)
            medium_volume_slope.iloc[i] = slope_volume_medium
    
    # Calculate combined price momentum (weighted average)
    price_momentum = 0.6 * medium_price_slope + 0.4 * short_price_slope
    
    # Calculate combined volume momentum (weighted average)
    volume_momentum = 0.6 * medium_volume_slope + 0.4 * short_volume_slope
    
    # Detect divergences and generate factor
    for i in range(len(data)):
        if i >= 20:
            # Check if price and volume trends are aligned
            price_dir = 1 if price_momentum.iloc[i] > 0 else -1
            volume_dir = 1 if volume_momentum.iloc[i] > 0 else -1
            
            # Calculate divergence magnitude
            divergence_mag = abs(price_momentum.iloc[i] - volume_momentum.iloc[i])
            
            # Generate factor: price momentum multiplied by volume momentum sign
            # Scaled by divergence magnitude to emphasize significant divergences
            if volume_dir != 0:
                factor_values.iloc[i] = (price_momentum.iloc[i] * volume_dir) * divergence_mag
            else:
                factor_values.iloc[i] = price_momentum.iloc[i] * divergence_mag
    
    # Normalize the factor values
    if len(factor_values.dropna()) > 0:
        factor_values = (factor_values - factor_values.mean()) / factor_values.std()
    
    return factor_values
