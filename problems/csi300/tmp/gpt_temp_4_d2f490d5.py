import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Adaptive Multi-Timeframe Price-Volume Divergence Alpha Factor
    
    This factor captures price-volume divergences across multiple timeframes
    with continuous adaptive weighting based on volatility regimes.
    """
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Initialize EMA variables
    ema_fast_p = df['close'].copy()
    ema_med_p = df['close'].copy()
    ema_slow_p = df['close'].copy()
    
    ema_fast_v = df['volume'].copy()
    ema_med_v = df['volume'].copy()
    ema_slow_v = df['volume'].copy()
    
    ema_range = (df['high'] - df['low']).copy()
    ema_vol_chg = abs(df['volume'].diff()).copy()
    
    # Calculate EMAs
    for i in range(1, len(df)):
        # Price EMAs
        ema_fast_p.iloc[i] = 0.5 * df['close'].iloc[i] + 0.5 * ema_fast_p.iloc[i-1]
        ema_med_p.iloc[i] = 0.25 * df['close'].iloc[i] + 0.75 * ema_med_p.iloc[i-1]
        ema_slow_p.iloc[i] = 0.125 * df['close'].iloc[i] + 0.875 * ema_slow_p.iloc[i-1]
        
        # Volume EMAs
        ema_fast_v.iloc[i] = 0.5 * df['volume'].iloc[i] + 0.5 * ema_fast_v.iloc[i-1]
        ema_med_v.iloc[i] = 0.25 * df['volume'].iloc[i] + 0.75 * ema_med_v.iloc[i-1]
        ema_slow_v.iloc[i] = 0.125 * df['volume'].iloc[i] + 0.875 * ema_slow_v.iloc[i-1]
        
        # Volatility EMAs
        daily_range = df['high'].iloc[i] - df['low'].iloc[i]
        ema_range.iloc[i] = 0.33 * daily_range + 0.67 * ema_range.iloc[i-1]
        
        vol_change = abs(df['volume'].iloc[i] - df['volume'].iloc[i-1])
        ema_vol_chg.iloc[i] = 0.33 * vol_change + 0.67 * ema_vol_chg.iloc[i-1]
    
    # Calculate price momentum
    price_mom_fast = ema_fast_p - ema_fast_p.shift(3)
    price_mom_med = ema_med_p - ema_med_p.shift(7)
    price_mom_slow = ema_slow_p - ema_slow_p.shift(14)
    
    # Calculate volume momentum
    volume_mom_fast = ema_fast_v - ema_fast_v.shift(3)
    volume_mom_med = ema_med_v - ema_med_v.shift(7)
    volume_mom_slow = ema_slow_v - ema_slow_v.shift(14)
    
    # Volatility scaling
    price_vol_scale = 1 / (ema_range + 0.0001)
    volume_vol_scale = 1 / (ema_vol_chg + 0.0001)
    
    # Direct divergence construction
    fast_divergence = (price_mom_fast * price_vol_scale) - (volume_mom_fast * volume_vol_scale)
    medium_divergence = (price_mom_med * price_vol_scale) - (volume_mom_med * volume_vol_scale)
    slow_divergence = (price_mom_slow * price_vol_scale) - (volume_mom_slow * volume_vol_scale)
    
    # Volatility regime signal
    current_vol = ema_range
    baseline_vol = ema_range.shift(3)
    vol_ratio = current_vol / (baseline_vol + 0.0001)
    
    # Continuous weight functions
    fast_weight = np.maximum(0, np.minimum(1, 2.5 - vol_ratio))
    medium_weight = np.maximum(0, np.minimum(1, 1.5 - abs(1.5 - vol_ratio)))
    slow_weight = np.maximum(0, np.minimum(1, vol_ratio - 0.5))
    
    # Alpha construction
    fast_component = fast_divergence * fast_weight
    medium_component = medium_divergence * medium_weight
    slow_component = slow_divergence * slow_weight
    
    final_alpha = fast_component + medium_component + slow_component
    
    return final_alpha
