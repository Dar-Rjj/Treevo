import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Regime-Aware Momentum Convergence with Volume Acceleration
    Combines price momentum convergence with volume acceleration patterns,
    uses adaptive smoothing based on market conditions, and applies
    regime-aware scaling without normalization for enhanced stability
    """
    # Price momentum convergence components
    short_momentum = df['close'] / df['close'].shift(3) - 1
    medium_momentum = df['close'] / df['close'].shift(5) - 1
    momentum_convergence = short_momentum - medium_momentum
    
    # Volume acceleration metrics
    volume_accel = (df['volume'] / df['volume'].shift(3)) / (df['volume'].shift(3) / df['volume'].shift(6) + 1e-7) - 1
    amount_accel = (df['amount'] / df['amount'].shift(3)) / (df['amount'].shift(3) / df['amount'].shift(6) + 1e-7) - 1
    
    # Volume-price alignment
    price_volume_alignment = np.sign(momentum_convergence) * (volume_accel + amount_accel)
    
    # Intraday strength indicators
    intraday_strength = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-7)
    range_efficiency = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-7)
    
    # Adaptive smoothing based on market volatility
    price_range = df['high'] - df['low']
    vol_regime = price_range.rolling(window=10, min_periods=1).std()
    
    # Regime-aware window selection
    fast_window = np.where(vol_regime > vol_regime.rolling(window=20, min_periods=1).mean(), 3, 5)
    slow_window = np.where(vol_regime > vol_regime.rolling(window=20, min_periods=1).mean(), 8, 13)
    
    # Apply adaptive smoothing
    smoothed_momentum = pd.Series(np.nan, index=df.index)
    smoothed_volume = pd.Series(np.nan, index=df.index)
    
    for i in range(len(df)):
        if i >= max(slow_window):
            win_fast = int(fast_window[i])
            win_slow = int(slow_window[i])
            smoothed_momentum.iloc[i] = momentum_convergence.iloc[i-win_fast+1:i+1].mean()
            smoothed_volume.iloc[i] = price_volume_alignment.iloc[i-win_slow+1:i+1].mean()
    
    # Combined factor with regime scaling
    momentum_component = smoothed_momentum * (1 + smoothed_volume)
    efficiency_component = intraday_strength * range_efficiency
    
    # Final factor without normalization
    factor = momentum_component * (1 + efficiency_component)
    
    return factor
