import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Momentum acceleration: 2-day vs 5-day momentum difference for trend acceleration
    momentum_2d = (df['close'] - df['close'].shift(2)) / df['close'].shift(2)
    momentum_5d = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    momentum_acceleration = momentum_2d - momentum_5d
    
    # Volume confirmation: volume trend aligned with price momentum
    volume_trend = (df['volume'] - df['volume'].shift(3)) / df['volume'].shift(3)
    volume_confirmation = np.sign(momentum_2d) * volume_trend
    
    # Volatility normalization: recent volatility scaling
    recent_volatility = df['close'].pct_change().rolling(window=5).std()
    volatility_normalized_momentum = momentum_2d / (recent_volatility + 1e-7)
    
    # Regime-aware scaling: market state adjustment using rolling percentiles
    momentum_regime = momentum_2d.rolling(window=20).apply(lambda x: (x.iloc[-1] - x.quantile(0.5)) / (x.quantile(0.75) - x.quantile(0.25) + 1e-7))
    
    # Core factor: multiplicative blend with bounded components
    core_factor = (
        np.tanh(momentum_acceleration) *  # Bounded acceleration
        np.tanh(volume_confirmation) *    # Bounded volume confirmation
        np.tanh(volatility_normalized_momentum) *  # Bounded normalized momentum
        np.tanh(momentum_regime)          # Bounded regime adjustment
    )
    
    return core_factor
