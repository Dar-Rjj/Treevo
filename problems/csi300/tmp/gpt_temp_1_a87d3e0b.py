import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate daily volatility using High-Low range
    daily_vol = (df['high'] - df['low']) / df['low']
    
    # 20-day volatility (rolling standard deviation of daily volatility)
    vol_20d = daily_vol.rolling(window=20, min_periods=10).std()
    
    # 60-day volatility percentiles for regime classification
    vol_60d_percentile = vol_20d.rolling(window=60, min_periods=30).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    # Classify volatility regimes
    high_vol_regime = vol_60d_percentile > 0.8
    low_vol_regime = vol_60d_percentile < 0.2
    
    # Calculate returns
    ret_3d = df['close'].pct_change(3)
    ret_10d = df['close'].pct_change(10)
    
    # High volatility regime: oversold bounce signals
    oversold_condition = ret_3d < -0.08
    oversold_magnitude = -ret_3d  # Magnitude of oversold condition
    
    # Low volatility regime: momentum breakdown signals
    momentum_breakdown_condition = (ret_10d > 0) & (ret_3d < 0)
    momentum_reversal_ratio = -ret_3d / (ret_10d + 1e-8)  # Negative 3-day vs positive 10-day
    
    # Liquidity signals
    volume_20d_median = df['volume'].rolling(window=20, min_periods=10).median()
    volume_surge = df['volume'] / (volume_20d_median + 1e-8)
    
    # Price-volume efficiency: volume per unit price movement
    price_range = df['high'] - df['low']
    volume_efficiency = df['volume'] / (price_range + 1e-8)
    efficiency_20d_median = volume_efficiency.rolling(window=20, min_periods=10).median()
    efficiency_ratio = volume_efficiency / (efficiency_20d_median + 1e-8)
    
    # Generate composite signals by regime
    high_vol_signal = oversold_magnitude * volume_surge
    low_vol_signal = momentum_reversal_ratio * efficiency_ratio
    
    # Combine signals based on volatility regime
    factor = pd.Series(index=df.index, dtype=float)
    factor[high_vol_regime] = high_vol_signal[high_vol_regime]
    factor[low_vol_regime] = low_vol_signal[low_vol_regime]
    
    # For normal volatility periods, use weighted average
    normal_regime = ~high_vol_regime & ~low_vol_regime
    factor[normal_regime] = (high_vol_signal[normal_regime] * 0.3 + 
                           low_vol_signal[normal_regime] * 0.7)
    
    # Fill NaN values with 0
    factor = factor.fillna(0)
    
    return factor
