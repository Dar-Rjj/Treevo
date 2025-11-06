import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Velocity Divergence Analysis
    # Short-term Velocity: (Close_t-1 - Close_t-3) / Close_t-3
    short_term_velocity = (data['close'].shift(1) - data['close'].shift(3)) / data['close'].shift(3)
    
    # Medium-term Velocity: (Close_t-1 - Close_t-8) / Close_t-8
    medium_term_velocity = (data['close'].shift(1) - data['close'].shift(8)) / data['close'].shift(8)
    
    # Velocity Divergence: Short-term - Medium-term velocity
    velocity_divergence = short_term_velocity - medium_term_velocity
    
    # Efficiency Assessment
    # Range Efficiency: |Close_t-1 - Close_t-2| / (High_t-1 - Low_t-1)
    range_efficiency = (abs(data['close'].shift(1) - data['close'].shift(2)) / 
                       (data['high'].shift(1) - data['low'].shift(1)))
    
    # Volatility Persistence: 5-day ATR / 10-day ATR
    def calculate_atr(window):
        high_low = data['high'] - data['low']
        high_close_prev = abs(data['high'] - data['close'].shift(1))
        low_close_prev = abs(data['low'] - data['close'].shift(1))
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        return true_range.rolling(window=window, min_periods=1).mean()
    
    atr_5 = calculate_atr(5)
    atr_10 = calculate_atr(10)
    volatility_persistence = atr_5 / atr_10
    
    # Volume Confirmation
    # Volume Acceleration: (3-day volume average / 8-day volume average) - 1
    volume_3d_avg = data['volume'].rolling(window=3, min_periods=1).mean()
    volume_8d_avg = data['volume'].rolling(window=8, min_periods=1).mean()
    volume_acceleration = (volume_3d_avg / volume_8d_avg) - 1
    
    # Volume Trend Slope: Linear slope of 5-day volume series
    def volume_slope(series):
        if len(series) < 2:
            return 0
        x = np.arange(len(series))
        slope = np.polyfit(x, series, 1)[0]
        return slope
    
    volume_trend_slope = data['volume'].rolling(window=5, min_periods=2).apply(
        volume_slope, raw=False
    )
    
    # Signal Integration
    # Core Signal: Velocity Divergence × Range Efficiency × Volatility Persistence
    core_signal = velocity_divergence * range_efficiency * volatility_persistence
    
    # Final Alpha: Core Signal × Volume Acceleration × Volume Trend Slope
    alpha = core_signal * volume_acceleration * volume_trend_slope
    
    return alpha
