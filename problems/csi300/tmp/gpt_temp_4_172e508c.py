import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Momentum acceleration with regime-aware volatility scaling and volume-confirmed efficiency
    # Uses nonlinear transforms for robustness and captures price-volume trend alignment
    
    # Calculate momentum acceleration (change in momentum trend)
    momentum_3d = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    momentum_6d = (df['close'] - df['close'].shift(6)) / df['close'].shift(6)
    momentum_acceleration = momentum_6d - momentum_3d.rolling(window=3).mean()
    
    # Calculate regime-aware volatility using high-low range (more robust than returns std)
    daily_range_pct = (df['high'] - df['low']) / df['close']
    volatility_regime = daily_range_pct.rolling(window=5).mean()
    
    # Volume trend confirmation with nonlinear scaling
    volume_trend = df['volume'].rolling(window=5).apply(
        lambda x: np.sign(x[-1] - x[0]) * np.log1p(abs(x[-1] - x[0])) if len(x) == 5 else np.nan
    )
    
    # Price efficiency with smoothing and nonlinear transform
    price_efficiency = abs(df['close'].pct_change()) / (daily_range_pct + 1e-7)
    smoothed_efficiency = price_efficiency.rolling(window=3).apply(
        lambda x: np.tanh(np.mean(x)) if len(x) == 3 else np.nan
    )
    
    # Combine factors with regime-aware scaling and volume confirmation
    volatility_scaled_momentum = momentum_acceleration / (volatility_regime + 1e-7)
    volume_confirmation = 1 + np.tanh(volume_trend)
    alpha_factor = volatility_scaled_momentum * volume_confirmation * smoothed_efficiency
    
    return alpha_factor
