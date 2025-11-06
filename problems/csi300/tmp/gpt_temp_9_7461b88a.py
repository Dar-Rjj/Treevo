import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Volume-synchronized momentum divergence with volatility regime adaptation.
    
    Interpretation:
    - Combines momentum divergence across multiple timeframes (intraday, overnight, daily)
    - Synchronizes momentum signals with volume pressure to confirm strength
    - Adapts to volatility regimes using rolling volatility percentiles
    - Uses momentum acceleration to capture changing trend dynamics
    - Volume divergence helps identify institutional vs. retail participation
    - Positive values indicate strong bullish momentum with volume confirmation
    - Negative values suggest bearish pressure with volume distribution
    """
    
    # Momentum components
    intraday_return = (df['close'] - df['open']) / df['open']
    overnight_return = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    daily_return = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Momentum divergence - difference between intraday and overnight momentum
    momentum_divergence = intraday_return - overnight_return
    
    # Momentum acceleration - change in daily momentum
    momentum_acceleration = daily_return - daily_return.shift(1)
    
    # Volume pressure and divergence
    volume_ma_5 = df['volume'].rolling(window=5).mean()
    volume_pressure = df['volume'] / (volume_ma_5 + 1e-7)
    volume_divergence = (df['volume'] - volume_ma_5) / (volume_ma_5 + 1e-7)
    
    # Volatility regime using rolling percentiles
    daily_range = (df['high'] - df['low']) / df['open']
    vol_10d_percentile = daily_range.rolling(window=20).apply(
        lambda x: (x.iloc[-1] - x.quantile(0.3)) / (x.quantile(0.7) - x.quantile(0.3) + 1e-7)
    )
    
    # Regime weights based on volatility percentile
    high_vol_weight = (vol_10d_percentile > 0.6).astype(float)
    low_vol_weight = (vol_10d_percentile < 0.4).astype(float)
    medium_vol_weight = 1 - high_vol_weight - low_vol_weight
    
    # Volume-synchronized momentum
    volume_sync_momentum = momentum_divergence * np.sign(volume_pressure - 1)
    
    # Combined factor with regime adaptation
    alpha_factor = (
        # Base momentum in medium volatility
        medium_vol_weight * (momentum_divergence + 0.3 * momentum_acceleration) +
        # Enhanced acceleration focus in high volatility
        high_vol_weight * (0.4 * momentum_divergence + 0.6 * momentum_acceleration) +
        # Volume-confirmed signals in low volatility
        low_vol_weight * (volume_sync_momentum + 0.2 * volume_divergence) +
        # Volume-pressure multiplier across all regimes
        0.15 * volume_pressure * momentum_divergence
    )
    
    return alpha_factor
