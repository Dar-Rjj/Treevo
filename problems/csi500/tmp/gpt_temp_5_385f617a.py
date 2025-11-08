import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Momentum-Volume Composite factor
    Combines multi-timeframe momentum with volume confirmation in a regime-adaptive framework
    """
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate True Range for volatility regime detection
    high_low = df['high'] - df['low']
    high_close_prev = abs(df['high'] - df['close'].shift(1))
    low_close_prev = abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    
    # Multi-Timeframe Momentum Components
    # Short-term momentum (3-day)
    mom_short_raw = df['close'] / df['close'].shift(3) - 1
    vol_regime_short = (true_range.rolling(window=10).apply(
        lambda x: (pd.Series(x).rank(pct=True).iloc[-1] if len(x) == 10 else np.nan), raw=False
    ) / 100) ** 0.5
    mom_short = mom_short_raw * vol_regime_short
    
    # Medium-term momentum (8-day)
    mom_medium_raw = df['close'] / df['close'].shift(8) - 1
    vol_regime_medium = (true_range.rolling(window=15).apply(
        lambda x: (pd.Series(x).rank(pct=True).iloc[-1] if len(x) == 15 else np.nan), raw=False
    ) / 100) ** 0.5
    mom_medium = mom_medium_raw * vol_regime_medium
    
    # Long-term momentum (20-day)
    mom_long_raw = df['close'] / df['close'].shift(20) - 1
    vol_regime_long = (true_range.rolling(window=30).apply(
        lambda x: (pd.Series(x).rank(pct=True).iloc[-1] if len(x) == 30 else np.nan), raw=False
    ) / 100) ** 0.5
    mom_long = mom_long_raw * vol_regime_long
    
    # Multiplicative Volume Confirmation
    # Volume momentum components
    vol_mom_short = df['volume'] / df['volume'].shift(3) - 1
    vol_mom_medium = df['volume'] / df['volume'].shift(8) - 1
    vol_mom_long = df['volume'] / df['volume'].shift(20) - 1
    
    # Volume regime composite with cube root transform
    vol_regime_short_15d = (vol_mom_short.rolling(window=15).apply(
        lambda x: (pd.Series(x).rank(pct=True).iloc[-1] if len(x) == 15 else np.nan), raw=False
    ) / 100) ** (1/3)
    
    vol_regime_medium_15d = (vol_mom_medium.rolling(window=15).apply(
        lambda x: (pd.Series(x).rank(pct=True).iloc[-1] if len(x) == 15 else np.nan), raw=False
    ) / 100) ** (1/3)
    
    vol_regime_long_15d = (vol_mom_long.rolling(window=15).apply(
        lambda x: (pd.Series(x).rank(pct=True).iloc[-1] if len(x) == 15 else np.nan), raw=False
    ) / 100) ** (1/3)
    
    # Multiplicative combination of volume regimes
    volume_composite = (vol_regime_short_15d * vol_regime_medium_15d * vol_regime_long_15d) ** (1/3)
    
    # Adaptive Factor Integration
    # Momentum composite with sign preservation
    momentum_sign = np.sign(mom_short * mom_medium * mom_long)
    momentum_magnitude = (abs(mom_short * mom_medium * mom_long)) ** (1/3)
    momentum_composite = momentum_sign * momentum_magnitude
    
    # Regime-weighted final factor with hyperbolic tangent transform
    regime_weighted_factor = momentum_composite * volume_composite
    final_factor = np.tanh(regime_weighted_factor * 2)  # Scaling factor for appropriate bounds
    
    return final_factor
