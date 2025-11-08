import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Compressed Momentum Divergence factor
    Detects momentum divergence during compressed volatility periods
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Detect Compressed Volatility Periods
    # Calculate daily range percentage
    daily_range_pct = (data['high'] - data['low']) / data['low']
    
    # Identify compression periods using rolling percentiles (20-day window)
    range_20d_percentile = daily_range_pct.rolling(window=20).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5
    )
    
    # Compression intensity (inverse of percentile - lower percentile = more compression)
    compression_intensity = 1 - range_20d_percentile
    
    # 2. Calculate Momentum Divergence in Compression
    # Multiple momentum periods
    momentum_3d = data['close'].pct_change(periods=3)
    momentum_8d = data['close'].pct_change(periods=8)
    momentum_13d = data['close'].pct_change(periods=13)
    
    # Momentum divergence - difference between short and medium-term momentum
    mom_divergence_short = momentum_3d - momentum_8d
    mom_divergence_medium = momentum_8d - momentum_13d
    
    # Combined divergence score
    momentum_divergence = mom_divergence_short * mom_divergence_medium
    
    # 3. Generate Compression-Break Signal
    # Weight divergence by compression intensity
    compression_weighted_divergence = momentum_divergence * compression_intensity
    
    # Volume expansion confirmation (5-day volume ratio)
    volume_5d_ratio = data['volume'] / data['volume'].rolling(window=5).mean()
    
    # Breakout magnitude relative to compressed range
    range_breakout_ratio = daily_range_pct / daily_range_pct.rolling(window=20).mean()
    
    # Final factor calculation
    factor = (
        compression_weighted_divergence * 
        volume_5d_ratio * 
        range_breakout_ratio
    )
    
    return factor
