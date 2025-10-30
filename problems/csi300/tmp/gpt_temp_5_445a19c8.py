import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    # Calculate Price Trend Acceleration
    # Short-term momentum: 5-day minus 2-day return
    short_term_momentum = df['close'] / df['close'].shift(5) - df['close'] / df['close'].shift(2)
    
    # Medium-term momentum: 10-day minus 5-day return
    medium_term_momentum = df['close'] / df['close'].shift(10) - df['close'] / df['close'].shift(5)
    
    # Price acceleration: difference between medium and short term momentum
    price_acceleration = medium_term_momentum - short_term_momentum
    
    # Calculate Volume Trend using linear regression slope
    def volume_slope(series):
        if len(series) < 5:
            return np.nan
        x = np.arange(len(series))
        slope, _, _, _, _ = stats.linregress(x, series.values)
        return slope
    
    volume_trend = df['volume'].rolling(window=5, min_periods=5).apply(volume_slope, raw=False)
    
    # Combine Acceleration and Volume Signals
    acceleration_volume_signal = price_acceleration * volume_trend
    
    # Calculate Recent Volatility using Average True Range
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_10 = true_range.rolling(window=10, min_periods=10).mean()
    
    # Scale signal by volatility
    scaled_signal = acceleration_volume_signal / atr_10
    
    # Apply Mean Reversion Logic
    # Calculate 30-day rolling Z-score using local statistics
    rolling_mean = scaled_signal.rolling(window=30, min_periods=30).mean()
    rolling_std = scaled_signal.rolling(window=30, min_periods=30).std()
    z_score = (scaled_signal - rolling_mean) / rolling_std
    
    # Invert signal when at extremes (beyond ±2 standard deviations)
    final_signal = scaled_signal.copy()
    extreme_high = z_score > 2
    extreme_low = z_score < -2
    final_signal[extreme_high] = -scaled_signal[extreme_high]
    final_signal[extreme_low] = -scaled_signal[extreme_low]
    
    # Apply smoothing with 3-day moving average
    smoothed_signal = final_signal.rolling(window=3, min_periods=3).mean()
    
    return smoothed_signal
