import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Compute Gap Momentum
    # Daily gap: (Close - Open) / Open
    daily_gap = (data['close'] - data['open']) / data['open']
    
    # Gap acceleration: gap_t - gap_{t-1}
    gap_acceleration = daily_gap.diff()
    
    # 3-day gap momentum: sum(gap acceleration over 3 days)
    gap_momentum = gap_acceleration.rolling(window=3, min_periods=1).sum()
    
    # Volume Confirmation
    # Volume surge: Volume > 1.5 * 10-day volume median
    volume_median_10d = data['volume'].rolling(window=10, min_periods=1).median()
    volume_surge = (data['volume'] > 1.5 * volume_median_10d).astype(float)
    
    # Volume trend: 5-day volume slope
    def calc_slope(series):
        if len(series) < 2:
            return 0
        x = np.arange(len(series))
        return np.polyfit(x, series, 1)[0] / np.mean(series) if np.mean(series) != 0 else 0
    
    volume_trend = data['volume'].rolling(window=5, min_periods=1).apply(calc_slope, raw=False)
    
    # Volume-gap correlation: 8-day correlation(gap changes, volume changes)
    gap_changes = daily_gap.diff()
    volume_changes = data['volume'].pct_change()
    volume_gap_corr = gap_changes.rolling(window=8, min_periods=1).corr(volume_changes).fillna(0)
    
    # Range Efficiency Analysis
    # Daily range: (High - Low) / Open
    daily_range = (data['high'] - data['low']) / data['open']
    
    # Close position: (Close - Low) / (High - Low)
    high_low_range = data['high'] - data['low']
    close_position = (data['close'] - data['low']) / np.where(high_low_range != 0, high_low_range, 1)
    
    # Efficiency trend: 5-day slope of |Close position - 0.5|
    efficiency_measure = np.abs(close_position - 0.5)
    efficiency_trend = efficiency_measure.rolling(window=5, min_periods=1).apply(calc_slope, raw=False)
    
    # Generate Composite Factor
    # Base signal: Gap momentum × Volume surge indicator
    base_signal = gap_momentum * volume_surge
    
    # Volume adjustment: × (1 + Volume trend) × (1 + Volume-gap correlation)
    volume_adjustment = (1 + volume_trend) * (1 + volume_gap_corr)
    
    # Range filter: × (2 - |Efficiency trend|)
    range_filter = 2 - np.abs(efficiency_trend)
    
    # Final factor: Base × Volume adjustment × Range filter
    final_factor = base_signal * volume_adjustment * range_filter
    
    return final_factor
