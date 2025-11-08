import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize result series
    factor_values = pd.Series(index=data.index, dtype=float)
    
    # Calculate linear regression slope helper function
    def calc_slope(series):
        if len(series) < 2:
            return np.nan
        x = np.arange(len(series))
        slope, _, _, _, _ = linregress(x, series)
        return slope
    
    # Calculate rolling average volume for scaling
    data['avg_volume_20'] = data['volume'].rolling(window=20, min_periods=1).mean()
    
    for i in range(len(data)):
        if i < 20:  # Need at least 20 periods for medium-term trend
            factor_values.iloc[i] = 0.0
            continue
            
        current_data = data.iloc[:i+1]
        
        # Calculate Short-Term Trend (t-5 to t)
        if i >= 5:
            short_term_prices = current_data['close'].iloc[i-5:i+1].values
            short_term_trend = calc_slope(short_term_prices)
        else:
            short_term_trend = np.nan
        
        # Calculate Medium-Term Trend (t-20 to t)
        medium_term_prices = current_data['close'].iloc[i-20:i+1].values
        medium_term_trend = calc_slope(medium_term_prices)
        
        # Calculate Acceleration Ratio
        if not np.isnan(short_term_trend) and medium_term_trend != 0:
            acceleration_ratio = short_term_trend / medium_term_trend
            acceleration = acceleration_ratio - 1
        else:
            acceleration = 0.0
        
        # Calculate Volume Trend (t-5 to t)
        if i >= 5:
            volume_data = current_data['volume'].iloc[i-5:i+1].values
            volume_trend = calc_slope(volume_data)
        else:
            volume_trend = 0.0
        
        # Volume-Price Alignment
        if acceleration > 0 and volume_trend > 0:
            aligned_acceleration = acceleration
        elif acceleration < 0 and volume_trend < 0:
            aligned_acceleration = acceleration
        else:
            aligned_acceleration = acceleration * -0.5  # Penalty for misalignment
        
        # Final Factor Value
        current_volume = data['volume'].iloc[i]
        avg_volume = data['avg_volume_20'].iloc[i]
        
        if avg_volume > 0:
            factor_value = aligned_acceleration * current_volume / avg_volume
        else:
            factor_value = aligned_acceleration * current_volume
        
        factor_values.iloc[i] = factor_value
    
    # Fill any remaining NaN values
    factor_values = factor_values.fillna(0.0)
    
    return factor_values
