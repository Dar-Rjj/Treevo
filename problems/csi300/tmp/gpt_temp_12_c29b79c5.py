import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    """
    Dynamic Price-Volume Divergence with Acceleration Momentum factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate Price Acceleration
    data['price_change_1'] = data['close'].diff(1)
    data['price_change_2'] = data['price_change_1'].diff(1)
    data['price_acceleration'] = data['price_change_2']
    
    # Calculate Volume Divergence using linear regression slope
    def volume_slope(volume_series):
        if len(volume_series) < 2:
            return 0
        x = np.arange(len(volume_series))
        slope, _, _, _, _ = stats.linregress(x, volume_series)
        return slope
    
    data['volume_slope'] = data['volume'].rolling(window=10, min_periods=3).apply(
        volume_slope, raw=False
    )
    
    # Calculate Price-Volume Divergence
    data['price_volume_divergence'] = data['price_acceleration'] * data['volume_slope']
    
    # Calculate Range Efficiency
    data['efficiency_ratio'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['efficiency_ratio'] = data['efficiency_ratio'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Apply Momentum Persistence Filter
    def persistence_multiplier(acceleration_series):
        if len(acceleration_series) < 3:
            return 1.0
        # Check if acceleration maintains direction for 3 days
        recent_acc = acceleration_series.iloc[-3:]
        if len(recent_acc) < 3:
            return 1.0
        # Check if all have same sign
        if (recent_acc > 0).all() or (recent_acc < 0).all():
            return 1.5  # Boost if persistent
        return 1.0
    
    data['persistence_multiplier'] = data['price_acceleration'].rolling(window=3, min_periods=1).apply(
        lambda x: persistence_multiplier(pd.Series(x)), raw=False
    )
    
    # Combine all components
    data['factor'] = (
        data['price_volume_divergence'] * 
        data['efficiency_ratio'] * 
        data['persistence_multiplier']
    )
    
    # Clean and return the factor
    factor = data['factor'].replace([np.inf, -np.inf], 0).fillna(0)
    return factor
