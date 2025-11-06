import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate directional range efficiencies
    high_low_range = data['high'] - data['low']
    high_low_range = high_low_range.replace(0, np.nan)  # Avoid division by zero
    
    upward_efficiency = (data['high'] - data['open']) / high_low_range
    downward_efficiency = (data['open'] - data['low']) / high_low_range
    
    # Efficiency momentum - 3-day correlation between upward and downward efficiencies
    efficiency_corr = upward_efficiency.rolling(window=3, min_periods=2).corr(downward_efficiency)
    
    # Efficiency trend using linear regression slope over 5 days
    def calc_slope(series):
        if len(series) < 2:
            return np.nan
        x = np.arange(len(series))
        return stats.linregress(x, series.values)[0]
    
    upward_trend = upward_efficiency.rolling(window=5, min_periods=3).apply(calc_slope, raw=False)
    downward_trend = downward_efficiency.rolling(window=5, min_periods=3).apply(calc_slope, raw=False)
    
    efficiency_momentum = (upward_trend - downward_trend) * efficiency_corr
    
    # Volatility adjustment using intraday true range
    true_range = data['high'] - data['low']
    volatility_adjusted = efficiency_momentum / true_range.replace(0, np.nan)
    
    # Volume-validated divergence
    volume_acceleration = data['volume'] / data['volume'].shift(1) - 1
    
    # Volume-weighted efficiency persistence
    volume_weighted_efficiency = (upward_efficiency * data['volume']).rolling(window=3).mean() / data['volume'].rolling(window=3).mean()
    
    # Final divergence signal with volume confirmation
    volume_confirmation = np.where(volume_acceleration > 0, 1 + volume_acceleration, 1)
    divergence_signal = volatility_adjusted * volume_weighted_efficiency * volume_confirmation
    
    # Handle any remaining NaN values
    result = divergence_signal.fillna(0)
    
    return result
