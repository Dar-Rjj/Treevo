import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    # Need at least 10 days of data for calculations
    if len(df) < 10:
        return alpha
    
    # Calculate Price Acceleration
    close = df['close']
    
    # 5-day price return
    ret_5d = close / close.shift(5) - 1
    
    # 10-day price return  
    ret_10d = close / close.shift(10) - 1
    
    # Price acceleration signal
    price_acceleration = (ret_5d - ret_10d) * ret_5d
    
    # Calculate Volume Confirmation Pattern
    volume = df['volume']
    
    def calculate_volume_slope(volume_series, window):
        slopes = pd.Series(index=volume_series.index, dtype=float)
        for i in range(window-1, len(volume_series)):
            if i >= window-1:
                window_data = volume_series.iloc[i-window+1:i+1]
                if len(window_data) == window:
                    x = np.arange(window)
                    slope, _, _, _, _ = linregress(x, window_data.values)
                    slopes.iloc[i] = slope
        return slopes
    
    # 5-day volume slope
    vol_slope_5d = calculate_volume_slope(volume, 5)
    
    # 10-day volume slope
    vol_slope_10d = calculate_volume_slope(volume, 10)
    
    # Volume confirmation score
    volume_confirmation = np.sign(vol_slope_5d * vol_slope_10d) * np.sqrt(np.abs(vol_slope_5d * vol_slope_10d))
    
    # Calculate Price-Volume Divergence
    high = df['high']
    low = df['low']
    
    # 5-day True Range Average
    def true_range(high, low, close_prev):
        return np.maximum(high, close_prev) - np.minimum(low, close_prev)
    
    tr_5d_avg = pd.Series(index=df.index, dtype=float)
    volume_per_unit_5d = pd.Series(index=df.index, dtype=float)
    
    for i in range(4, len(df)):
        if i >= 4:
            # Calculate True Range for each day in 5-day window
            tr_values = []
            for j in range(i-4, i+1):
                if j > 0:
                    tr_val = true_range(high.iloc[j], low.iloc[j], close.iloc[j-1])
                    tr_values.append(tr_val)
            
            if len(tr_values) == 5:
                tr_5d_avg.iloc[i] = np.mean(tr_values)
                # Volume per unit price movement (5-day average)
                vol_sum = volume.iloc[i-4:i+1].sum()
                volume_per_unit_5d.iloc[i] = vol_sum / tr_5d_avg.iloc[i] if tr_5d_avg.iloc[i] > 0 else 0
    
    # Current day's volume per unit movement
    current_volume_per_unit = volume / (high - low)
    current_volume_per_unit = current_volume_per_unit.replace([np.inf, -np.inf], 0)
    current_volume_per_unit = current_volume_per_unit.fillna(0)
    
    # Divergence signal
    divergence_signal = (current_volume_per_unit / volume_per_unit_5d) - 1
    divergence_signal = divergence_signal.replace([np.inf, -np.inf], 0)
    divergence_signal = divergence_signal.fillna(0)
    
    # Generate Composite Alpha Factor
    # Combine acceleration with volume confirmation
    combined_signal = price_acceleration * volume_confirmation
    
    # Apply cubic root to reduce extreme values
    combined_signal = np.sign(combined_signal) * np.power(np.abs(combined_signal), 1/3)
    
    # Adjust with price-volume divergence
    adjusted_signal = combined_signal * (1 + divergence_signal)
    
    # Apply hyperbolic tangent for bounded scaling
    final_alpha = np.tanh(adjusted_signal)
    
    # Fill the alpha series
    alpha = final_alpha
    
    return alpha
