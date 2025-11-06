import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Reversal-Pressure Divergence Analysis
    # Short-Term Reversal
    high_low_range = data['high'] - data['low']
    short_term_reversal = ((data['close'] - data['low']) / high_low_range.replace(0, np.nan)) - \
                         ((data['high'] - data['close']) / high_low_range.replace(0, np.nan))
    
    # Medium-Term Reversal
    rolling_low_5 = data['low'].rolling(window=5, min_periods=1).min()
    rolling_high_5 = data['high'].rolling(window=5, min_periods=1).max()
    range_5 = rolling_high_5 - rolling_low_5
    medium_term_reversal = ((data['close'] - rolling_low_5) / range_5.replace(0, np.nan)) - 0.5
    
    # Reversal Divergence
    reversal_divergence = short_term_reversal * medium_term_reversal
    
    # Volume-Intensity Divergence
    # Intensity Ratio
    close_open_abs = abs(data['close'] - data['open'])
    intensity_recent = close_open_abs.rolling(window=3, min_periods=1).sum()
    intensity_older = close_open_abs.shift(3).rolling(window=5, min_periods=1).sum()
    intensity_ratio = intensity_recent / intensity_older.replace(0, np.nan)
    
    # Volume Concentration
    volume_avg_5 = data['volume'].rolling(window=5, min_periods=1).mean()
    volume_concentration = data['volume'] / volume_avg_5.replace(0, np.nan)
    
    # Volume-Intensity Divergence
    volume_intensity_divergence = intensity_ratio * volume_concentration
    
    # Combined Reversal-Pressure
    reversal_pressure = np.cbrt(reversal_divergence * volume_intensity_divergence)
    
    # Gap-Fill Momentum Confirmation
    # Gap Absorption Efficiency
    gap_size = abs(data['open'] - data['close'].shift(1)) / data['close'].shift(1).replace(0, np.nan)
    fill_ratio = abs(data['close'] - data['open']) / abs(data['open'] - data['close'].shift(1)).replace(0, np.nan)
    gap_efficiency = gap_size * fill_ratio * np.sign(data['close'] - data['open'])
    
    # Momentum Persistence
    # Direction Consistency
    close_diff = data['close'].diff()
    current_direction = np.sign(close_diff)
    direction_consistency = pd.Series(index=data.index, dtype=float)
    
    for i in range(len(data)):
        if i >= 4:
            window_directions = [np.sign(data['close'].iloc[j] - data['close'].iloc[j-1]) 
                               for j in range(i-4, i+1)]
            consistency = sum(1 for d in window_directions if d == current_direction.iloc[i]) / 5
            direction_consistency.iloc[i] = consistency
        else:
            direction_consistency.iloc[i] = 0.5
    
    # Magnitude Ratio
    abs_close_diff = abs(close_diff)
    avg_magnitude_5 = abs_close_diff.rolling(window=5, min_periods=1).mean()
    magnitude_ratio = abs_close_diff / avg_magnitude_5.replace(0, np.nan)
    
    # Momentum Persistence
    momentum_persistence = direction_consistency * magnitude_ratio
    
    # Combined Gap-Fill Momentum
    gap_fill_momentum = np.sqrt(gap_efficiency * momentum_persistence)
    
    # Final Alpha Synthesis
    # Base Signal
    base_signal = reversal_pressure * gap_fill_momentum
    
    # Volatility Adjustment
    high_low_range_daily = data['high'] - data['low']
    vol_recent = high_low_range_daily.rolling(window=5, min_periods=1).sum()
    vol_older = high_low_range_daily.shift(5).rolling(window=5, min_periods=1).sum()
    vol_adjustment = vol_recent / vol_older.replace(0, np.nan)
    
    volatility_adjusted = base_signal * vol_adjustment
    
    # Trend Filter
    close_avg_5 = data['close'].rolling(window=5, min_periods=1).mean()
    trend_filter = np.where(data['close'] > close_avg_5, 1, -1)
    
    # Final Alpha
    final_alpha = volatility_adjusted * trend_filter
    
    return final_alpha
