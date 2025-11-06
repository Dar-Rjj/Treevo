import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate daily price range
    data['daily_range'] = data['high'] - data['low']
    
    # Calculate range expansion ratios
    data['range_5d_avg'] = data['daily_range'].rolling(window=5, min_periods=3).mean()
    data['range_10d_avg'] = data['daily_range'].rolling(window=10, min_periods=5).mean()
    data['range_expansion_5d'] = data['daily_range'] / data['range_5d_avg']
    data['range_expansion_10d'] = data['daily_range'] / data['range_10d_avg']
    
    # Calculate volume-to-range ratio
    data['volume_to_range'] = data['volume'] / (data['daily_range'] + 1e-8)
    
    # Calculate volume-to-range trend (5-day slope)
    def calculate_slope(series):
        if len(series) < 3:
            return np.nan
        x = np.arange(len(series))
        return stats.linregress(x, series.values)[0]
    
    data['volume_range_trend'] = data['volume_to_range'].rolling(window=5, min_periods=3).apply(calculate_slope, raw=False)
    
    # Volume confirmation: positive correlation between range expansion and volume
    data['range_volume_corr'] = data['daily_range'].rolling(window=5, min_periods=3).corr(data['volume'])
    
    # Calculate normalized position in range
    data['normalized_position'] = (data['close'] - data['low']) / (data['daily_range'] + 1e-8)
    
    # Calculate position stability (5-day standard deviation of normalized position)
    data['position_stability'] = data['normalized_position'].rolling(window=5, min_periods=3).std()
    
    # Combine range expansion with volume signals
    # Range expansion with positive volume correlation = bullish
    range_expansion_signal = (data['range_expansion_5d'] + data['range_expansion_10d']) / 2
    volume_confirmation = data['range_volume_corr'] * np.sign(data['volume_range_trend'])
    
    # Adjust for price position context
    # Weight signals higher when price is near range extremes
    position_weight = 1 + np.abs(data['normalized_position'] - 0.5) * 2
    
    # Create directional bias based on close position
    # Bullish bias when price closes in upper range, bearish when in lower range
    directional_bias = (data['normalized_position'] - 0.5) * 2
    
    # Combine all components
    # Range expansion with volume confirmation gets positive weight
    # Range contraction with volume divergence gets negative weight
    base_factor = range_expansion_signal * volume_confirmation
    adjusted_factor = base_factor * position_weight * directional_bias
    
    # Final factor: range-volume convergence with position adjustment
    alpha_factor = adjusted_factor.fillna(0)
    
    return alpha_factor
