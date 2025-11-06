import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Volatility-scaled momentum components
    # Short-term momentum (3-day) normalized by recent volatility (5-day)
    momentum_3d = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    volatility_5d = df['close'].pct_change().rolling(window=5).std()
    vol_scaled_momentum = momentum_3d / (volatility_5d + 1e-7)
    
    # Volume acceleration with regime detection
    # Volume momentum (3-day) vs longer-term volume trend (10-day)
    volume_momentum_3d = (df['volume'] - df['volume'].shift(3)) / df['volume'].shift(3)
    volume_trend_10d = df['volume'].rolling(window=10).mean() / df['volume'].shift(10) - 1
    volume_acceleration = volume_momentum_3d - volume_trend_10d
    
    # Regime shift detection using price and volume confluence
    # Price regime: momentum persistence
    momentum_persistence = (df['close'].pct_change(3).rolling(window=5).std() / 
                           df['close'].pct_change(3).rolling(window=20).std())
    
    # Volume regime: abnormal volume detection
    volume_zscore = (df['volume'] - df['volume'].rolling(window=20).mean()) / df['volume'].rolling(window=20).std()
    volume_regime = np.tanh(volume_zscore / 2)  # Smooth transition function
    
    # Adaptive bounds using rolling percentiles for robust scaling
    # Momentum bounds (20-day lookback)
    momentum_3d_rank = momentum_3d.rolling(window=20).apply(
        lambda x: (x.iloc[-1] - x.quantile(0.1)) / (x.quantile(0.9) - x.quantile(0.1) + 1e-7)
    )
    
    # Volume acceleration bounds (20-day lookback)
    volume_accel_rank = volume_acceleration.rolling(window=20).apply(
        lambda x: (x.iloc[-1] - x.quantile(0.1)) / (x.quantile(0.9) - x.quantile(0.1) + 1e-7)
    )
    
    # Multiplicative combination with regime-based weighting
    # Core momentum component with volatility scaling
    momentum_component = vol_scaled_momentum * (1 + np.tanh(momentum_3d_rank))
    
    # Volume component with acceleration and regime adjustment
    volume_component = (1 + volume_accel_rank) * (1 + volume_regime)
    
    # Regime shift component - captures momentum persistence changes
    regime_component = 1 + np.tanh(momentum_persistence - 1)
    
    # Final multiplicative combination with smooth transitions
    factor = (momentum_component * 
              volume_component * 
              regime_component * 
              (1 + np.tanh(volume_accel_rank))  # Additional volume acceleration emphasis
             )
    
    return factor
