import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Multi-Timeframe Momentum Divergence with Volume-Regime Adaptive Smoothing
    Combines short-term and medium-term momentum signals with volume trend confirmation,
    detects momentum divergences across timeframes, and applies regime-aware smoothing
    using adaptive windows for stable factor generation
    """
    # Multi-timeframe momentum signals
    short_term_momentum = df['close'] / df['close'].shift(3) - 1
    medium_term_momentum = df['close'] / df['close'].shift(10) - 1
    
    # Momentum divergence detection
    momentum_divergence = short_term_momentum - medium_term_momentum
    momentum_acceleration = short_term_momentum - short_term_momentum.shift(3)
    
    # Volume trend components
    volume_trend_short = df['volume'] / df['volume'].rolling(window=5, min_periods=1).mean() - 1
    volume_trend_medium = df['volume'] / df['volume'].rolling(window=15, min_periods=1).mean() - 1
    volume_divergence = volume_trend_short - volume_trend_medium
    
    # Volume-momentum synchronization
    volume_momentum_alignment = np.sign(short_term_momentum) * volume_trend_short
    volume_divergence_alignment = np.sign(momentum_divergence) * volume_divergence
    
    # Price range efficiency
    daily_range = (df['high'] - df['low']) / df['open']
    close_position = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-7)
    range_efficiency = (close_position - 0.5) * (1 / (daily_range + 1e-7))
    
    # Regime detection using volatility
    price_volatility = df['close'].rolling(window=20, min_periods=1).std() / df['close'].rolling(window=20, min_periods=1).mean()
    volume_volatility = df['volume'].rolling(window=20, min_periods=1).std() / df['volume'].rolling(window=20, min_periods=1).mean()
    high_vol_regime = (price_volatility > price_volatility.rolling(window=50, min_periods=1).quantile(0.7)) | \
                     (volume_volatility > volume_volatility.rolling(window=50, min_periods=1).quantile(0.7))
    
    # Adaptive smoothing windows based on regime
    smooth_window = np.where(high_vol_regime, 3, 8)
    
    # Core factor components
    momentum_component = momentum_divergence * (1 + momentum_acceleration)
    volume_component = volume_momentum_alignment + volume_divergence_alignment
    efficiency_component = range_efficiency * (1 + volume_trend_short)
    
    # Combined raw factor
    raw_factor = momentum_component + volume_component + efficiency_component
    
    # Apply regime-aware adaptive smoothing
    factor = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i < max(smooth_window):
            factor.iloc[i] = raw_factor.iloc[i]
        else:
            window = smooth_window[i]
            factor.iloc[i] = raw_factor.iloc[i-window+1:i+1].mean()
    
    return factor
