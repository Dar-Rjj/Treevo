import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Novel alpha factor: Multi-Timeframe Momentum-Volume-Range Synergy with Adaptive Bounding
    
    Economic intuition:
    - Multi-timeframe momentum captures both short-term and medium-term price dynamics
    - Volume-range interaction measures trading intensity relative to price movement scope
    - Directional gap strength incorporates overnight information and opening momentum
    - Rolling percentile features adapt to changing market conditions without normalization
    - Multiplicative synergy amplifies aligned signals across different market dimensions
    - Adaptive bounding prevents extreme values while preserving signal directionality
    """
    
    # Multi-timeframe bounded momentum
    momentum_1d = df['close'].pct_change(1)
    momentum_3d = df['close'].pct_change(3)
    momentum_5d = df['close'].pct_change(5)
    
    # Momentum acceleration and convergence
    momentum_accel = momentum_1d - momentum_3d
    momentum_convergence = momentum_3d - momentum_5d
    
    # Bounded momentum transforms with adaptive scaling
    momentum_short = np.tanh(momentum_1d * 8)
    momentum_medium = np.tanh(momentum_3d * 6)
    momentum_accel_bounded = np.tanh(momentum_accel * 5)
    momentum_conv_bounded = np.tanh(momentum_convergence * 4)
    
    # Combined momentum factor
    momentum_combined = momentum_short + momentum_medium + momentum_accel_bounded + momentum_conv_bounded
    
    # Volume-range intensity with rolling context
    price_range = df['high'] - df['low']
    range_ratio = price_range / df['close']
    volume_intensity = df['volume'] * range_ratio
    
    # Rolling volume momentum with bounded transform
    volume_ma_3 = volume_intensity.rolling(window=3).mean()
    volume_ma_5 = volume_intensity.rolling(window=5).mean()
    volume_momentum = (volume_ma_3 - volume_ma_5) / (volume_ma_5 + 1e-7)
    volume_bounded = np.tanh(volume_momentum * 6)
    
    # Directional gap strength with opening momentum
    gap_strength = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    intraday_strength = (df['close'] - df['open']) / df['open']
    directional_power = gap_strength + intraday_strength
    directional_bounded = np.tanh(directional_power * 10)
    
    # Rolling range regime detection
    range_rolling_rank = range_ratio.rolling(window=8).apply(lambda x: (x.rank().iloc[-1] - 1) / (len(x) - 1) if len(x) == 8 else np.nan)
    range_regime = np.tanh((range_rolling_rank - 0.5) * 3)
    
    # Volume volatility adaptation
    volume_volatility = df['volume'].rolling(window=5).std()
    volume_vol_rank = volume_volatility.rolling(window=8).apply(lambda x: (x.rank().iloc[-1] - 1) / (len(x) - 1) if len(x) == 8 else np.nan)
    volume_regime = np.tanh((volume_vol_rank - 0.5) * 2)
    
    # Multiplicative synergy core
    momentum_volume_synergy = momentum_combined * volume_bounded
    directional_enhanced = momentum_volume_synergy * directional_bounded
    regime_adapted = directional_enhanced * range_regime * volume_regime
    
    # Final bounded factor
    alpha_factor = np.tanh(regime_adapted * 2)
    
    return alpha_factor
