import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Multi-Timeframe Momentum Calculation
    close = df['close']
    
    # Short-term momentum (5-day)
    mom_short = (close - close.shift(5)) / close.shift(5)
    
    # Medium-term momentum (10-day)
    mom_medium = (close - close.shift(10)) / close.shift(10)
    
    # Long-term momentum (20-day)
    mom_long = (close - close.shift(20)) / close.shift(20)
    
    # Momentum Divergence Analysis
    short_medium_div = np.abs(mom_short - mom_medium)
    medium_long_div = np.abs(mom_medium - mom_long)
    
    # Overall divergence strength with momentum direction
    divergence_strength = (short_medium_div + medium_long_div) * np.sign(mom_short)
    
    # Volatility Scaling
    high = df['high']
    low = df['low']
    
    # Short-term volatility (5-day average daily range)
    daily_range_short = high.rolling(window=5).apply(lambda x: (x - low.loc[x.index]).mean(), raw=False)
    
    # Medium-term volatility (10-day average daily range)
    daily_range_medium = high.rolling(window=10).apply(lambda x: (x - low.loc[x.index]).mean(), raw=False)
    
    # Adaptive volatility adjustment
    volatility_ratio = daily_range_short / daily_range_medium
    volatility_ratio = volatility_ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    
    # Scale divergences by volatility
    scaled_short_div = short_medium_div / daily_range_short
    scaled_medium_div = medium_long_div / daily_range_medium
    
    # Volatility-scaled momentum divergence
    vol_scaled_divergence = (scaled_short_div + scaled_medium_div * volatility_ratio) * divergence_strength
    
    # Volume Trend Integration
    volume = df['volume']
    
    # Volume momentum calculation
    vol_mom_5d = (volume - volume.shift(5)) / volume.shift(5)
    vol_mom_10d = (volume - volume.shift(10)) / volume.shift(10)
    
    # Volume-price confirmation
    vol_price_5d = vol_mom_5d * mom_short
    vol_price_10d = vol_mom_10d * mom_medium
    volume_confirmation = (vol_price_5d + vol_price_10d) / 2
    
    # Volume divergence signal
    vol_div_signal = (vol_mom_5d - mom_short + vol_mom_10d - mom_medium) / 2
    
    # Volume-adjusted component
    volume_adjusted = vol_scaled_divergence * volume_confirmation + vol_div_signal
    
    # Final alpha factor with momentum direction preservation
    recent_price_action = (close - close.shift(3)) / close.shift(3)
    final_factor = volume_adjusted * np.sign(mom_short) * np.abs(recent_price_action)
    
    return final_factor
