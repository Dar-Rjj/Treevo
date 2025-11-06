import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price-Volume Divergence Momentum factor that combines price trend acceleration
    with volume anomaly detection, adjusted for volatility.
    """
    # Price Trend Component
    # High-Low Range Momentum
    recent_high = df['high'].rolling(window=5).max()
    recent_low = df['low'].rolling(window=5).min()
    recent_range = recent_high - recent_low
    
    previous_high = df['high'].shift(5).rolling(window=5).max()
    previous_low = df['low'].shift(5).rolling(window=5).min()
    previous_range = previous_high - previous_low
    
    range_ratio = recent_range / previous_range.replace(0, np.nan)
    
    # Close Price Acceleration
    recent_return = df['close'] / df['close'].shift(3) - 1
    previous_return = df['close'].shift(3) / df['close'].shift(6) - 1
    price_acceleration = recent_return - previous_return
    
    # Combine price components
    price_component = range_ratio * price_acceleration
    
    # Volume Anomaly Component
    # Volume Spike Detection
    volume_ma_20 = df['volume'].rolling(window=20).mean()
    volume_ratio = df['volume'] / volume_ma_20.replace(0, np.nan)
    
    # Volume Trend Consistency
    volume_slope_5 = df['volume'].rolling(window=5).apply(
        lambda x: np.polyfit(range(5), x, 1)[0] if not x.isna().any() else np.nan
    )
    volume_slope_10 = df['volume'].rolling(window=10).apply(
        lambda x: np.polyfit(range(10), x, 1)[0] if not x.isna().any() else np.nan
    )
    volume_consistency = volume_slope_5 / volume_slope_10.replace(0, np.nan)
    
    # Combine volume components
    volume_component = volume_ratio * volume_consistency
    
    # Divergence Combination
    raw_factor = price_component * volume_component
    
    # Sign Correction - Check Price-Volume Direction Alignment
    price_direction = np.sign(price_acceleration)
    volume_direction = np.sign(volume_slope_5)
    
    # Invert factor if price and volume are diverging (opposite directions)
    sign_correction = np.where(price_direction * volume_direction < 0, -1, 1)
    corrected_factor = raw_factor * sign_correction
    
    # Volatility Adjustment
    volatility_10 = df['high'].rolling(window=10).max() - df['low'].rolling(window=10).min()
    volatility_adjusted = corrected_factor / volatility_10.replace(0, np.nan)
    
    return volatility_adjusted
