import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Factor: Multi-Scale Volatility-Scaled Momentum with Volume-Confirmed Efficiency
    # Economic intuition: Combines multi-period momentum scaled by dynamic volatility, 
    # confirmed by volume trends and price efficiency across multiple timeframes
    
    # Multi-period momentum (1,3,5 days)
    mom_1d = df['close'] / df['close'].shift(1) - 1
    mom_3d = df['close'] / df['close'].shift(3) - 1
    mom_5d = df['close'] / df['close'].shift(5) - 1
    
    # Geometric mean of momentum for stability
    momentum_combined = (mom_1d * mom_3d * mom_5d).abs().pow(1/3) * np.sign(mom_3d)
    
    # Multi-scale volatility scaling
    daily_range = (df['high'] - df['low']) / df['close']
    vol_5d = daily_range.rolling(5).std()
    vol_10d = daily_range.rolling(10).std()
    
    # Combined volatility scaling (harmonic mean for robustness)
    vol_scaling = 2 / (1/(vol_5d + 1e-7) + 1/(vol_10d + 1e-7))
    vol_scaled_momentum = momentum_combined / vol_scaling
    
    # Volume acceleration with trend confirmation
    volume_accel_5d = df['volume'] / df['volume'].rolling(5).mean()
    volume_trend = df['volume'].rolling(5).apply(lambda x: np.polyfit(range(5), x, 1)[0])
    
    # Capped volume factor
    volume_capped = volume_accel_5d.clip(lower=0.5, upper=2.0)
    volume_trend_capped = volume_trend.clip(lower=-1000, upper=1000)
    volume_factor = volume_capped * (1 + volume_trend_capped / 1000)
    
    # Multi-window price efficiency
    efficiency_1d = (df['close'] - df['close'].shift(1)).abs() / (df['high'] - df['low'])
    efficiency_3d = efficiency_1d.rolling(3).mean()
    efficiency_5d = efficiency_1d.rolling(5).mean()
    
    # Combined efficiency (arithmetic mean)
    efficiency_combined = (efficiency_1d + efficiency_3d + efficiency_5d) / 3
    
    # Final factor: Volatility-scaled momentum × Volume confirmation × Price efficiency
    factor = vol_scaled_momentum * volume_factor * efficiency_combined
    
    return factor
