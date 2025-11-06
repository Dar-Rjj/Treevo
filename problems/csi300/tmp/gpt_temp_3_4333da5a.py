import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Novel alpha factor combining volatility-volume acceleration divergence with 
    price-volume fractal efficiency and trend fracture acceleration reversion.
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Volatility-Volume Acceleration Divergence
    # Calculate volatility acceleration using short-term (3d) vs medium-term (8d) High-Low ranges
    high_low_range = df['high'] - df['low']
    vol_accel_3d = high_low_range.rolling(window=3).mean() / high_low_range.rolling(window=6).mean() - 1
    vol_accel_8d = high_low_range.rolling(window=8).mean() / high_low_range.rolling(window=16).mean() - 1
    
    # Compute volume acceleration across matching time horizons
    vol_accel_3d_vol = df['volume'].rolling(window=3).mean() / df['volume'].rolling(window=6).mean() - 1
    vol_accel_8d_vol = df['volume'].rolling(window=8).mean() / df['volume'].rolling(window=16).mean() - 1
    
    # Detect acceleration divergence between volatility and volume regimes
    vol_vol_divergence = (vol_accel_3d - vol_accel_8d) - (vol_accel_3d_vol - vol_accel_8d_vol)
    
    # Apply regime-specific weighting based on divergence magnitude
    vol_div_weight = np.tanh(np.abs(vol_vol_divergence) * 2)
    
    # Price-Volume Fractal Efficiency
    # Calculate directional volume efficiency using Close-Open vs Volume acceleration
    price_change = (df['close'] - df['open']) / df['open']
    volume_efficiency = price_change / (df['volume'].rolling(window=3).std() + 1e-8)
    
    # Compute multi-scale volume persistence across different time horizons
    vol_persistence_3d = df['volume'].rolling(window=3).apply(lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) == 3 and not np.isnan(x).any() else 0)
    vol_persistence_8d = df['volume'].rolling(window=8).apply(lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) == 8 and not np.isnan(x).any() else 0)
    
    # Detect volume clustering patterns relative to price acceleration
    price_accel = (df['close'].pct_change(periods=1) - df['close'].pct_change(periods=3)).rolling(window=5).mean()
    vol_clustering = (vol_persistence_3d - vol_persistence_8d) * price_accel
    
    # Combine efficiency signals with volume acceleration divergence
    fractal_efficiency = volume_efficiency * vol_clustering * vol_div_weight
    
    # Trend Fracture Acceleration Reversion
    # Calculate trend strength using multi-timeframe price acceleration
    trend_3d = df['close'].rolling(window=3).mean().pct_change(periods=1)
    trend_8d = df['close'].rolling(window=8).mean().pct_change(periods=1)
    trend_21d = df['close'].rolling(window=21).mean().pct_change(periods=1)
    
    # Multi-timeframe acceleration
    accel_3d_8d = trend_3d - trend_8d
    accel_8d_21d = trend_8d - trend_21d
    
    # Identify trend fracture points through acceleration breakdown
    trend_fracture = (accel_3d_8d.rolling(window=5).std() + accel_8d_21d.rolling(window=5).std()) / 2
    
    # Compute reversion probability based on fracture magnitude and volume divergence
    volume_divergence = (df['volume'] - df['volume'].rolling(window=21).mean()) / df['volume'].rolling(window=21).std()
    reversion_prob = np.abs(trend_fracture) * np.abs(volume_divergence)
    
    # Apply dynamic reversion thresholds based on acceleration history
    reversion_threshold = reversion_prob.rolling(window=21).quantile(0.7)
    reversion_signal = (reversion_prob > reversion_threshold) * np.sign(trend_3d) * -1
    
    # Final alpha factor combining all components
    alpha_factor = (fractal_efficiency * vol_div_weight + reversion_signal * reversion_prob) / 2
    
    # Normalize the factor
    alpha_factor = (alpha_factor - alpha_factor.rolling(window=63).mean()) / alpha_factor.rolling(window=63).std()
    
    return alpha_factor
