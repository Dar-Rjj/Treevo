import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate Intraday Reversal Strength
    # Intraday Range
    data['intraday_range'] = (data['high'] - data['low']) / data['open']
    
    # Price Rejection
    data['hl_midpoint'] = (data['high'] + data['low']) / 2
    data['price_rejection'] = (data['close'] - data['hl_midpoint']) / (data['high'] - data['low'])
    
    # 3-day average of price rejection magnitude
    data['rejection_magnitude'] = data['price_rejection'].abs()
    data['reversal_signal'] = data['rejection_magnitude'].rolling(window=3, min_periods=1).mean()
    
    # Analyze Volume Acceleration Patterns
    # Volume Acceleration
    data['volume_5d_avg'] = data['volume'].rolling(window=5, min_periods=1).mean()
    data['volume_acceleration'] = data['volume'] / data['volume_5d_avg']
    
    # Volume Trend using 5-day linear regression slope
    def calc_volume_slope(volume_series):
        if len(volume_series) < 2:
            return 0
        x = np.arange(len(volume_series))
        slope, _, _, _, _ = linregress(x, volume_series)
        return slope
    
    data['volume_trend'] = data['volume'].rolling(window=5, min_periods=2).apply(
        calc_volume_slope, raw=False
    )
    
    # Abnormal Volume Spikes
    data['abnormal_volume'] = data['volume_acceleration'] > 2.0
    
    # Combine Reversal and Volume Signals
    # Strong price rejection filter
    strong_rejection = data['price_rejection'].abs() > 0.3
    strong_volume_accel = data['volume_acceleration'] > 1.5
    
    # Volume trend direction matching reversal direction
    volume_trend_match = (
        (data['volume_trend'] > 0) & (data['price_rejection'] > 0) |
        (data['volume_trend'] < 0) & (data['price_rejection'] < 0)
    )
    
    # Composite signal
    data['composite_signal'] = data['price_rejection'] * data['volume_acceleration']
    
    # Apply filters
    valid_signal_mask = strong_rejection & strong_volume_accel & volume_trend_match
    data['filtered_signal'] = data['composite_signal'].where(valid_signal_mask, 0)
    
    # Incorporate Range Breakout Context
    # Calculate True Range
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['close'].shift(1))
    data['tr3'] = abs(data['low'] - data['close'].shift(1))
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # 10-day Average True Range
    data['atr_10d'] = data['true_range'].rolling(window=10, min_periods=1).mean()
    
    # Range Breakouts
    data['range_breakout'] = data['intraday_range'] > (1.5 * data['atr_10d'])
    
    # Adjust Signal for Breakout Conditions
    # Enhance signal during range expansion, reduce during contraction
    data['range_adjustment'] = np.where(
        data['range_breakout'],
        data['intraday_range'] / data['atr_10d'],  # Enhance during expansion
        0.5  # Reduce during normal/contraction periods
    )
    
    # Final factor calculation
    data['factor'] = data['filtered_signal'] * data['range_adjustment']
    
    return data['factor']
