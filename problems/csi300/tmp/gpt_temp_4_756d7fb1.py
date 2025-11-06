import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    """
    Volume-Price Fractal Efficiency with Regime Switching alpha factor
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Calculate Fractal Efficiency
    # Price Path Length - sum of absolute movements over 10 days
    price_path_length = (data['high'] - data['low']).rolling(window=10, min_periods=10).sum()
    
    # Net Price Change - absolute difference between day t and day t-10
    net_price_change = (data['close'] - data['close'].shift(10)).abs()
    
    # Efficiency Ratio
    efficiency_ratio = net_price_change / price_path_length
    efficiency_ratio = efficiency_ratio.replace([np.inf, -np.inf], np.nan).fillna(0)
    efficiency_ratio = np.clip(efficiency_ratio, 0, 1)  # Bound between 0 and 1
    
    # 2. Incorporate Volume Confirmation
    # Calculate Volume Trend using linear regression slope over 10 days
    def volume_slope(volume_series):
        if len(volume_series) < 10:
            return np.nan
        x = np.arange(len(volume_series))
        slope, _, _, _, _ = stats.linregress(x, volume_series)
        return slope
    
    volume_trend = data['volume'].rolling(window=10, min_periods=10).apply(
        volume_slope, raw=False
    )
    
    # Normalize volume trend to [-1, 1] range
    volume_trend_normalized = volume_trend / (volume_trend.abs().rolling(window=50, min_periods=50).mean())
    volume_trend_normalized = np.clip(volume_trend_normalized, -1, 1)
    
    # Volume-Weighted Efficiency Signal
    volume_weighted_efficiency = efficiency_ratio * (1 + volume_trend_normalized)
    
    # 3. Implement Regime Detection
    # Calculate Volatility Clustering using High-Low range
    daily_range = data['high'] - data['low']
    volatility_20d = daily_range.rolling(window=20, min_periods=20).std()
    volatility_ma = volatility_20d.rolling(window=20, min_periods=20).mean()
    
    # Classify Regimes
    volatility_ratio = volatility_20d / volatility_ma
    
    # Regime classification
    high_vol_threshold = 1.2
    low_vol_threshold = 0.8
    
    regime = pd.Series(index=data.index, dtype='object')
    regime[volatility_ratio >= high_vol_threshold] = 'high_volatility'
    regime[volatility_ratio <= low_vol_threshold] = 'low_volatility'
    regime[regime.isna()] = 'transition'
    
    # 4. Apply Regime-Specific Adjustments
    final_factor = pd.Series(index=data.index, dtype=float)
    
    # High volatility regime: focus on volume confirmation
    high_vol_mask = regime == 'high_volatility'
    final_factor[high_vol_mask] = (
        volume_weighted_efficiency[high_vol_mask] * 
        (1 + volume_trend_normalized[high_vol_mask])
    )
    
    # Low volatility regime: emphasize efficiency signals
    low_vol_mask = regime == 'low_volatility'
    final_factor[low_vol_mask] = (
        efficiency_ratio[low_vol_mask] * 
        (1 + efficiency_ratio[low_vol_mask])
    )
    
    # Transition regime: blend both components
    trans_mask = regime == 'transition'
    final_factor[trans_mask] = (
        0.5 * efficiency_ratio[trans_mask] + 
        0.5 * volume_weighted_efficiency[trans_mask]
    )
    
    # Final normalization
    final_factor = (final_factor - final_factor.rolling(window=50, min_periods=50).mean()) / \
                   final_factor.rolling(window=50, min_periods=50).std()
    
    return final_factor.fillna(0)
