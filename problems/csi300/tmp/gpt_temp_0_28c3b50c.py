import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress, pearsonr

def heuristics_v2(df):
    """
    Price-Momentum and Volume Divergence Factor
    Combines short-term and medium-term price momentum with volume trends
    """
    # Create a copy to avoid modifying original dataframe
    data = df.copy()
    
    # Initialize result series
    factor_values = pd.Series(index=data.index, dtype=float)
    
    # Minimum required data points for calculations
    min_periods = 20
    
    for i in range(min_periods, len(data)):
        current_data = data.iloc[:i+1]
        
        # Price Momentum Component
        # Short-Term Price Trend (6 days)
        if i >= 5:
            short_term_prices = current_data['close'].iloc[i-5:i+1].values
            short_term_indices = np.arange(len(short_term_prices))
            short_term_slope, _, _, _, _ = linregress(short_term_indices, short_term_prices)
        else:
            short_term_slope = 0
        
        # Medium-Term Price Trend (20 days)
        if i >= 19:
            medium_term_indices = [i-19, i-14, i-9, i-4, i]
            medium_term_prices = current_data['close'].iloc[medium_term_indices].values
            medium_term_x = np.arange(len(medium_term_prices))
            medium_term_slope, _, _, _, _ = linregress(medium_term_x, medium_term_prices)
        else:
            medium_term_slope = 0
        
        # Volume Divergence Component
        # Volume Trend (6 days)
        if i >= 5:
            short_term_volumes = current_data['volume'].iloc[i-5:i+1].values
            short_term_vol_indices = np.arange(len(short_term_volumes))
            volume_slope, _, _, _, _ = linregress(short_term_vol_indices, short_term_volumes)
        else:
            volume_slope = 0
        
        # Volume-Price Correlation (10 days)
        if i >= 9:
            recent_prices = current_data['close'].iloc[i-9:i+1].values
            recent_volumes = current_data['volume'].iloc[i-9:i+1].values
            correlation, _ = pearsonr(recent_prices, recent_volumes)
            # Handle NaN correlation (can occur with constant values)
            if np.isnan(correlation):
                correlation = 0
        else:
            correlation = 0
        
        # Combine Components
        # Multiply Short-Term Price Trend by Volume Trend
        short_term_component = short_term_slope * volume_slope
        
        # Multiply Medium-Term Price Trend by Volume-Price Correlation
        medium_term_component = medium_term_slope * correlation
        
        # Sum both weighted components
        factor_value = short_term_component + medium_term_component
        
        factor_values.iloc[i] = factor_value
    
    # Fill early values with 0
    factor_values = factor_values.fillna(0)
    
    return factor_values
