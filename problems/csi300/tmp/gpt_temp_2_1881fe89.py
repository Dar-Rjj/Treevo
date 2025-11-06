import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate Price Acceleration
    # First-Order Momentum
    ret_5d = data['close'].pct_change(5)
    ret_10d = data['close'].pct_change(10)
    
    # Second-Order Acceleration
    acceleration = ret_5d - ret_10d
    acceleration_change = acceleration - acceleration.shift(3)
    
    # Analyze Volume Pattern Divergence
    # Calculate Volume Trend using linear regression slopes
    def calculate_slope(series, window):
        slopes = pd.Series(index=series.index, dtype=float)
        for i in range(window-1, len(series)):
            if i >= window-1:
                y = series.iloc[i-window+1:i+1].values
                x = np.arange(len(y))
                slope, _, _, _, _ = stats.linregress(x, y)
                slopes.iloc[i] = slope
        return slopes
    
    volume_slope_5d = calculate_slope(data['volume'], 5)
    volume_slope_10d = calculate_slope(data['volume'], 10)
    
    # Identify Volume-Price Divergence
    price_accel_sign = np.sign(acceleration)
    volume_trend_sign = np.sign(volume_slope_5d)
    
    # Calculate divergence strength
    divergence_strength = np.abs(acceleration * volume_slope_5d)
    
    # Combine Signals with Market Regime
    # Detect Market Volatility Regime
    daily_range = (data['high'] - data['low']) / data['close']
    volatility_20d = daily_range.rolling(window=20, min_periods=10).mean()
    
    # Calculate historical volatility levels for classification
    volatility_60d_hist = volatility_20d.rolling(window=60, min_periods=30).quantile(0.7)
    
    # Classify regime (0 for low, 1 for high volatility)
    regime_score = (volatility_20d > volatility_60d_hist).astype(float)
    
    # Weight Signals by Regime
    # Adjust acceleration signal for volatility
    volatility_adj = np.where(volatility_20d > 0, 1 / volatility_20d, 1)
    regime_adjusted_acceleration = acceleration_change * regime_score * volatility_adj
    
    # Combine with volume divergence
    final_factor = regime_adjusted_acceleration * divergence_strength * price_accel_sign
    
    return final_factor
