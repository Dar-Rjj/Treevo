import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Enhanced alpha factor: Persistent volume trends with volatility-normalized momentum 
    across multiple timeframes using multiplicative alignment and adaptive thresholds.
    
    Factor interpretation:
    - Persistent volume trends: Uses expanding window volume ratios to capture sustained 
      directional volume changes rather than simple averages
    - Volatility-normalized momentum: Normalizes momentum signals by their respective 
      volatility regimes across 5-day, 15-day, and 30-day timeframes
    - Multiplicative alignment: Combines clean components multiplicatively to emphasize 
      signal coherence while reducing noise
    - Adaptive thresholds: Uses rolling percentiles to dynamically adjust signal strength
      based on recent market behavior
    - The factor targets stocks with persistent volume support, consistent momentum across
      timeframes in appropriate volatility contexts, and strong signal alignment
    """
    
    # Persistent volume trends: expanding window volume ratios
    volume_5d = df['volume'].rolling(window=5).mean()
    volume_10d = df['volume'].rolling(window=10).mean()
    volume_20d = df['volume'].rolling(window=20).mean()
    
    # Volume persistence ratios with expanding windows
    volume_persistence_short = volume_5d / volume_10d
    volume_persistence_medium = volume_10d / volume_20d
    volume_persistence_combined = volume_persistence_short * volume_persistence_medium
    
    # Multi-timeframe momentum with volatility normalization
    momentum_5d = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    momentum_15d = (df['close'] - df['close'].shift(15)) / df['close'].shift(15)
    momentum_30d = (df['close'] - df['close'].shift(30)) / df['close'].shift(30)
    
    # Volatility normalization for each momentum timeframe
    volatility_5d = (df['high'] - df['low']).rolling(window=5).std() / df['close'].rolling(window=5).mean()
    volatility_15d = (df['high'] - df['low']).rolling(window=15).std() / df['close'].rolling(window=15).mean()
    volatility_30d = (df['high'] - df['low']).rolling(window=30).std() / df['close'].rolling(window=30).mean()
    
    # Volatility-normalized momentum signals
    momentum_normalized_5d = momentum_5d / (volatility_5d + 1e-7)
    momentum_normalized_15d = momentum_15d / (volatility_15d + 1e-7)
    momentum_normalized_30d = momentum_30d / (volatility_30d + 1e-7)
    
    # Momentum alignment across timeframes
    momentum_alignment = momentum_normalized_5d * momentum_normalized_15d * momentum_normalized_30d
    
    # Price efficiency with volatility adjustment
    daily_range_efficiency = abs(df['close'] - df['close'].shift(1)) / ((df['high'] - df['low']) + 1e-7)
    efficiency_volatility_adj = daily_range_efficiency.rolling(window=10).mean() / (volatility_15d + 1e-7)
    
    # Adaptive thresholds using rolling percentiles
    volume_threshold = volume_persistence_combined.rolling(window=30).apply(lambda x: x.quantile(0.7))
    momentum_threshold = momentum_alignment.rolling(window=30).apply(lambda x: x.quantile(0.6))
    
    # Signal components with adaptive thresholding
    volume_signal = volume_persistence_combined > volume_threshold
    momentum_signal = momentum_alignment > momentum_threshold
    
    # Combined factor with multiplicative alignment of clean components
    alpha_factor = (momentum_alignment * volume_persistence_combined * efficiency_volatility_adj * 
                   volume_signal.astype(float) * momentum_signal.astype(float))
    
    return alpha_factor
