import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Adaptive Multi-Timeframe Momentum Divergence Alpha Factor
    
    This factor combines price and volume momentum across multiple timeframes,
    adaptively weighting them based on current volatility conditions to capture
    price-volume divergences that may predict future returns.
    """
    
    # Initialize result series
    result = pd.Series(index=data.index, dtype=float)
    
    # Extract price and volume data
    close = data['close']
    high = data['high']
    low = data['low']
    volume = data['volume']
    
    # Initialize EMA series
    # Price EMAs
    ema_fast_price = pd.Series(index=data.index, dtype=float)
    ema_medium_price = pd.Series(index=data.index, dtype=float)
    ema_slow_price = pd.Series(index=data.index, dtype=float)
    
    # Volume EMAs
    ema_fast_vol = pd.Series(index=data.index, dtype=float)
    ema_medium_vol = pd.Series(index=data.index, dtype=float)
    ema_slow_vol = pd.Series(index=data.index, dtype=float)
    
    # Range EMA
    daily_range = high - low
    ema_range = pd.Series(index=data.index, dtype=float)
    
    # Initialize first values
    if len(data) > 0:
        ema_fast_price.iloc[0] = close.iloc[0]
        ema_medium_price.iloc[0] = close.iloc[0]
        ema_slow_price.iloc[0] = close.iloc[0]
        ema_fast_vol.iloc[0] = volume.iloc[0]
        ema_medium_vol.iloc[0] = volume.iloc[0]
        ema_slow_vol.iloc[0] = volume.iloc[0]
        ema_range.iloc[0] = daily_range.iloc[0]
    
    # Calculate EMAs
    for i in range(1, len(data)):
        # Price EMAs
        ema_fast_price.iloc[i] = 0.3 * close.iloc[i] + 0.7 * ema_fast_price.iloc[i-1]
        ema_medium_price.iloc[i] = 0.1 * close.iloc[i] + 0.9 * ema_medium_price.iloc[i-1]
        ema_slow_price.iloc[i] = 0.05 * close.iloc[i] + 0.95 * ema_slow_price.iloc[i-1]
        
        # Volume EMAs
        ema_fast_vol.iloc[i] = 0.3 * volume.iloc[i] + 0.7 * ema_fast_vol.iloc[i-1]
        ema_medium_vol.iloc[i] = 0.1 * volume.iloc[i] + 0.9 * ema_medium_vol.iloc[i-1]
        ema_slow_vol.iloc[i] = 0.05 * volume.iloc[i] + 0.95 * ema_slow_vol.iloc[i-1]
        
        # Range EMA
        ema_range.iloc[i] = 0.15 * daily_range.iloc[i] + 0.85 * ema_range.iloc[i-1]
    
    # Calculate volatility scale factor
    volatility_scale = 1 / (ema_range + 0.0001)
    
    # Multi-timeframe momentum construction
    # Price momentum divergences
    price_fast_medium = (ema_fast_price - ema_medium_price) * volatility_scale
    price_medium_slow = (ema_medium_price - ema_slow_price) * volatility_scale
    price_fast_slow = (ema_fast_price - ema_slow_price) * volatility_scale
    
    # Volume momentum divergences
    volume_fast_medium = (ema_fast_vol - ema_medium_vol) * volatility_scale
    volume_medium_slow = (ema_medium_vol - ema_slow_vol) * volatility_scale
    volume_fast_slow = (ema_fast_vol - ema_slow_vol) * volatility_scale
    
    # Price-volume divergence signals
    fast_divergence = price_fast_medium - volume_fast_medium
    medium_divergence = price_medium_slow - volume_medium_slow
    slow_divergence = price_fast_slow - volume_fast_slow
    
    # Adaptive volatility weighting
    # Calculate volatility ratio (current vs baseline 10 days ago)
    baseline_volatility = ema_range.shift(10)
    volatility_ratio = ema_range / (baseline_volatility + 0.0001)
    
    # Timeframe-specific weights
    fast_weight = np.maximum(0, 1.5 - volatility_ratio)  # Emphasized in low volatility
    medium_weight = 1 - np.abs(1 - volatility_ratio)     # Balanced in normal volatility
    slow_weight = np.maximum(0, volatility_ratio - 0.5)  # Emphasized in high volatility
    
    # Final alpha calculation
    fast_component = fast_divergence * fast_weight
    medium_component = medium_divergence * medium_weight
    slow_component = slow_divergence * slow_weight
    
    # Sum all weighted components
    result = fast_component + medium_component + slow_component
    
    return result
