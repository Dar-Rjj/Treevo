import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Volatility-Scaled Price-Volume Momentum Divergence Alpha Factor
    
    Parameters:
    df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume', 'amount']
    
    Returns:
    Series: Alpha factor values indexed by date
    """
    
    # Extract price and volume data
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # Initialize all EMA series
    ema_price_mom_fast = pd.Series(index=close.index, dtype=float)
    ema_price_mom_med = pd.Series(index=close.index, dtype=float)
    ema_price_mom_slow = pd.Series(index=close.index, dtype=float)
    
    ema_volume_mom_fast = pd.Series(index=close.index, dtype=float)
    ema_volume_mom_med = pd.Series(index=close.index, dtype=float)
    ema_volume_mom_slow = pd.Series(index=close.index, dtype=float)
    
    ema_range = pd.Series(index=close.index, dtype=float)
    ema_vol_chg = pd.Series(index=close.index, dtype=float)
    
    # Calculate daily range
    daily_range = high - low
    
    # Calculate volume change
    volume_change = volume.diff().abs()
    
    # Initialize first values
    ema_price_mom_fast.iloc[0] = 0
    ema_price_mom_med.iloc[0] = 0
    ema_price_mom_slow.iloc[0] = 0
    
    ema_volume_mom_fast.iloc[0] = 0
    ema_volume_mom_med.iloc[0] = 0
    ema_volume_mom_slow.iloc[0] = 0
    
    ema_range.iloc[0] = daily_range.iloc[0]
    ema_vol_chg.iloc[0] = volume_change.iloc[0] if not np.isnan(volume_change.iloc[0]) else 0
    
    # Calculate EMA series iteratively
    for i in range(1, len(close)):
        # Price momentum calculations
        if i >= 5:
            price_mom_fast = close.iloc[i] - close.iloc[i-5]
            ema_price_mom_fast.iloc[i] = 0.4 * price_mom_fast + 0.6 * ema_price_mom_fast.iloc[i-1]
        else:
            ema_price_mom_fast.iloc[i] = ema_price_mom_fast.iloc[i-1]
            
        if i >= 10:
            price_mom_med = close.iloc[i] - close.iloc[i-10]
            ema_price_mom_med.iloc[i] = 0.2 * price_mom_med + 0.8 * ema_price_mom_med.iloc[i-1]
        else:
            ema_price_mom_med.iloc[i] = ema_price_mom_med.iloc[i-1]
            
        if i >= 20:
            price_mom_slow = close.iloc[i] - close.iloc[i-20]
            ema_price_mom_slow.iloc[i] = 0.1 * price_mom_slow + 0.9 * ema_price_mom_slow.iloc[i-1]
        else:
            ema_price_mom_slow.iloc[i] = ema_price_mom_slow.iloc[i-1]
        
        # Volume momentum calculations
        if i >= 5:
            volume_mom_fast = volume.iloc[i] - volume.iloc[i-5]
            ema_volume_mom_fast.iloc[i] = 0.4 * volume_mom_fast + 0.6 * ema_volume_mom_fast.iloc[i-1]
        else:
            ema_volume_mom_fast.iloc[i] = ema_volume_mom_fast.iloc[i-1]
            
        if i >= 10:
            volume_mom_med = volume.iloc[i] - volume.iloc[i-10]
            ema_volume_mom_med.iloc[i] = 0.2 * volume_mom_med + 0.8 * ema_volume_mom_med.iloc[i-1]
        else:
            ema_volume_mom_med.iloc[i] = ema_volume_mom_med.iloc[i-1]
            
        if i >= 20:
            volume_mom_slow = volume.iloc[i] - volume.iloc[i-20]
            ema_volume_mom_slow.iloc[i] = 0.1 * volume_mom_slow + 0.9 * ema_volume_mom_slow.iloc[i-1]
        else:
            ema_volume_mom_slow.iloc[i] = ema_volume_mom_slow.iloc[i-1]
        
        # Volatility scaling calculations
        ema_range.iloc[i] = 0.2 * daily_range.iloc[i] + 0.8 * ema_range.iloc[i-1]
        
        if not np.isnan(volume_change.iloc[i]):
            ema_vol_chg.iloc[i] = 0.2 * volume_change.iloc[i] + 0.8 * ema_vol_chg.iloc[i-1]
        else:
            ema_vol_chg.iloc[i] = ema_vol_chg.iloc[i-1]
    
    # Volatility scaling factors
    price_vol_scale = 1 / (ema_range + 0.0001)
    volume_vol_scale = 1 / (ema_vol_chg + 0.0001)
    
    # Volatility-scaled momentum components
    price_mom_fast_scaled = ema_price_mom_fast * price_vol_scale
    price_mom_med_scaled = ema_price_mom_med * price_vol_scale
    price_mom_slow_scaled = ema_price_mom_slow * price_vol_scale
    
    volume_mom_fast_scaled = ema_volume_mom_fast * volume_vol_scale
    volume_mom_med_scaled = ema_volume_mom_med * volume_vol_scale
    volume_mom_slow_scaled = ema_volume_mom_slow * volume_vol_scale
    
    # Price-volume divergence components
    fast_divergence = price_mom_fast_scaled - volume_mom_fast_scaled
    medium_divergence = price_mom_med_scaled - volume_mom_med_scaled
    slow_divergence = price_mom_slow_scaled - volume_mom_slow_scaled
    
    # Dynamic timeframe weighting based on volatility regime
    current_vol = ema_range
    reference_vol = ema_range.shift(10).fillna(ema_range)
    vol_ratio = current_vol / (reference_vol + 0.0001)
    
    fast_weight = np.maximum(0, 2 - vol_ratio)
    medium_weight = 1 - np.abs(1 - vol_ratio)
    slow_weight = np.maximum(0, vol_ratio - 1)
    
    # Final alpha construction
    fast_component = fast_divergence * fast_weight
    medium_component = medium_divergence * medium_weight
    slow_component = slow_divergence * slow_weight
    
    alpha = fast_component + medium_component + slow_component
    
    return alpha
