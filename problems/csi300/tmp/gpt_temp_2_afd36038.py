import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Hierarchical Price-Volume Divergence Factor
    Combines price and volume trend divergences with dynamic volatility and volume stability weighting
    """
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # Initialize result series
    factor = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df)):
        if i < 9:  # Need at least 10 days for medium-term trend
            factor.iloc[i] = 0
            continue
            
        current_idx = df.index[i]
        
        # 1. Compute Price Trend Divergence
        # Short-term price trend (t-4 to t)
        short_price_window = close.iloc[i-4:i+1]
        if len(short_price_window) >= 2:
            short_price_trend = np.polyfit(range(len(short_price_window)), short_price_window.values, 1)[0]
        else:
            short_price_trend = 0
            
        # Medium-term price trend (t-9 to t)
        medium_price_window = close.iloc[i-9:i+1]
        if len(medium_price_window) >= 2:
            medium_price_trend = np.polyfit(range(len(medium_price_window)), medium_price_window.values, 1)[0]
        else:
            medium_price_trend = 0
            
        # 2. Compute Volume Trend Divergence
        # Short-term volume trend (t-4 to t)
        short_volume_window = volume.iloc[i-4:i+1]
        if len(short_volume_window) >= 2:
            short_volume_trend = np.polyfit(range(len(short_volume_window)), short_volume_window.values, 1)[0]
        else:
            short_volume_trend = 0
            
        # Medium-term volume trend (t-9 to t)
        medium_volume_window = volume.iloc[i-9:i+1]
        if len(medium_volume_window) >= 2:
            medium_volume_trend = np.polyfit(range(len(medium_volume_window)), medium_volume_window.values, 1)[0]
        else:
            medium_volume_trend = 0
            
        # 3. Calculate Divergence Components
        # Price Trend Ratio
        if medium_price_trend != 0:
            price_ratio = short_price_trend / medium_price_trend - 1
        else:
            price_ratio = 0
            
        # Volume Trend Ratio
        if medium_volume_trend != 0:
            volume_ratio = short_volume_trend / medium_volume_trend - 1
        else:
            volume_ratio = 0
            
        # Cross-Divergence
        cross_divergence = price_ratio * volume_ratio
        cross_divergence_magnitude = np.sqrt(np.abs(cross_divergence)) if cross_divergence != 0 else 0
        cross_divergence_sign = np.sign(cross_divergence) if cross_divergence != 0 else 0
        
        # 4. Apply Dynamic Weighting
        # Volatility Adjustment - 5-day average daily range
        if i >= 4:
            daily_ranges = []
            for j in range(i-4, i+1):
                daily_range = high.iloc[j] - low.iloc[j]
                daily_ranges.append(daily_range)
            avg_daily_range = np.mean(daily_ranges) if daily_ranges else 1
        else:
            avg_daily_range = 1
            
        # Volume Stability - inverse of 5-day volume variance
        if i >= 4:
            volume_window = volume.iloc[i-4:i+1]
            volume_variance = np.var(volume_window) if len(volume_window) >= 2 else 1
            if volume_variance > 0:
                volume_stability = np.log(1 / volume_variance)
            else:
                volume_stability = 0
        else:
            volume_stability = 0
            
        # 5. Combine Final Factor
        divergence_component = cross_divergence_magnitude * cross_divergence_sign
        volatility_adjusted = divergence_component * avg_daily_range
        final_factor = volatility_adjusted * volume_stability
        
        factor.iloc[i] = final_factor
    
    return factor
