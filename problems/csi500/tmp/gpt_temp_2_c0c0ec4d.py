import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    for i in range(2, len(df)):
        # Current day data
        high_t = df['high'].iloc[i]
        low_t = df['low'].iloc[i]
        close_t = df['close'].iloc[i]
        open_t = df['open'].iloc[i]
        volume_t = df['volume'].iloc[i]
        
        # Previous day data
        high_t1 = df['high'].iloc[i-1]
        high_t2 = df['high'].iloc[i-2]
        low_t1 = df['low'].iloc[i-1]
        low_t2 = df['low'].iloc[i-2]
        close_t1 = df['close'].iloc[i-1]
        close_t2 = df['close'].iloc[i-2]
        volume_t1 = df['volume'].iloc[i-1]
        
        # Volume-Fractal Alignment
        high_volume_fractal_break = (high_t - max(high_t1, high_t2)) * volume_t
        low_volume_fractal_support = (low_t - min(low_t1, low_t2)) * volume_t
        fractal_volume_symmetry = high_volume_fractal_break + low_volume_fractal_support
        
        # Efficiency-Momentum Structure
        range_eff = ((close_t - open_t) / (high_t - low_t)) * volume_t if (high_t - low_t) != 0 else 0
        mom_accel = ((close_t / close_t1 - 1) / (close_t1 / close_t2 - 1)) if (close_t1 / close_t2 - 1) != 0 else 0
        
        # Fractal Momentum Divergence
        price_fractal_mom = (high_t - max(high_t1, high_t2)) * (close_t / close_t1 - 1)
        volume_fractal_mom = volume_t * fractal_volume_symmetry
        fractal_mom_divergence = price_fractal_mom - volume_fractal_mom
        
        # Regime Transition
        volume_accel = (volume_t / volume_t1 - 1) if volume_t1 != 0 else 0
        price_volume_regime_shift = volume_accel * (close_t / close_t1 - 1)
        
        # Alpha Construction
        raw_alpha = fractal_mom_divergence * mom_accel
        enhanced_alpha = raw_alpha * price_volume_regime_shift
        volatility_scaled_alpha = enhanced_alpha / (high_t - low_t) if (high_t - low_t) != 0 else 0
        
        result.iloc[i] = volatility_scaled_alpha
    
    return result
