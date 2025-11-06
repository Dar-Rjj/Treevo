import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    """
    Momentum-Adjusted Range-Volume Divergence factor
    Combines range-based momentum signals with volume divergence patterns
    """
    # Create copy to avoid modifying original data
    data = df.copy()
    
    # 1. Compute Daily Price Range
    data['daily_range'] = data['high'] - data['low']
    
    # 2. Compute Range Momentum Signals
    # 5-day Range Return
    data['range_5d_return'] = (data['daily_range'] / data['daily_range'].shift(5)) - 1
    
    # 10-day Range Return
    data['range_10d_return'] = (data['daily_range'] / data['daily_range'].shift(10)) - 1
    
    # Range Change Ratio
    data['range_change_ratio'] = data['daily_range'] / data['daily_range'].shift(1)
    
    # Range Direction Consistency (3-day window)
    range_direction = np.sign(data['daily_range'] - data['daily_range'].shift(1))
    data['range_dir_consistency'] = range_direction.rolling(window=3, min_periods=1).apply(
        lambda x: len(set(x.dropna())) if len(x.dropna()) > 0 else 1
    )
    # Convert to consistency score (higher = more consistent)
    data['range_dir_consistency'] = 1 / data['range_dir_consistency']
    
    # 3. Compute Volume Divergence Patterns
    # Volume Ratio (5-day vs 10-day mean)
    data['volume_5d_mean'] = data['volume'].rolling(window=5, min_periods=3).mean()
    data['volume_10d_mean'] = data['volume'].rolling(window=10, min_periods=5).mean()
    data['volume_ratio'] = data['volume_5d_mean'] / data['volume_10d_mean']
    
    # Volume Trend Slope (5-day linear regression)
    def volume_slope(x):
        if len(x) < 2:
            return 0
        return stats.linregress(range(len(x)), x)[0]
    
    data['volume_trend_slope'] = data['volume'].rolling(window=5, min_periods=3).apply(
        volume_slope, raw=True
    )
    
    # Volume Stability (Coefficient of Variation)
    data['volume_cv'] = data['volume'].rolling(window=10, min_periods=5).std() / \
                       data['volume'].rolling(window=10, min_periods=5).mean()
    data['volume_stability'] = 1 / (1 + data['volume_cv'])  # Higher = more stable
    
    # 4. Combine Range and Volume Signals
    # Primary Momentum Divergence components
    data['signal_5d_range_vol'] = data['range_5d_return'] * data['volume_ratio']
    data['signal_10d_range_vol'] = data['range_10d_return'] * data['volume_ratio']
    data['signal_range_change_vol'] = data['range_change_ratio'] * data['volume_trend_slope']
    
    # Persistence Adjustment
    # Weight signals by range direction consistency
    data['adj_signal_5d'] = data['signal_5d_range_vol'] * data['range_dir_consistency']
    data['adj_signal_10d'] = data['signal_10d_range_vol'] * data['range_dir_consistency']
    data['adj_signal_change'] = data['signal_range_change_vol'] * data['range_dir_consistency']
    
    # Apply Volume Stability filter
    stability_weight = data['volume_stability'].fillna(0.5)
    data['final_signal_5d'] = data['adj_signal_5d'] * stability_weight
    data['final_signal_10d'] = data['adj_signal_10d'] * stability_weight
    data['final_signal_change'] = data['adj_signal_change'] * stability_weight
    
    # 5. Generate Final Alpha Factor
    # Composite weighted average with range momentum as primary driver
    weights = np.array([0.4, 0.35, 0.25])  # Emphasize range momentum
    signals = np.column_stack([
        data['final_signal_5d'].fillna(0),
        data['final_signal_10d'].fillna(0),
        data['final_signal_change'].fillna(0)
    ])
    
    # Incorporate volume divergence direction
    volume_divergence = np.sign(data['volume_ratio'] - 1)
    alpha_factor = np.sum(signals * weights, axis=1) * volume_divergence
    
    # Return as pandas Series
    return pd.Series(alpha_factor, index=data.index, name='momentum_range_volume_divergence')
