import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Ultra-short momentum reversal with volatility-normalized volume acceleration
    # Uses regime-adaptive smoothing and thresholds for stability
    # Synergistic interaction between price, volume, and volatility components
    
    # Ultra-short momentum reversal (1-day vs 3-day momentum divergence)
    momentum_1d = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    momentum_3d = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    momentum_reversal = momentum_1d - momentum_3d
    
    # Volume acceleration (rate of change in volume momentum)
    volume_roc = df['volume'] / df['volume'].shift(1) - 1
    volume_acceleration = volume_roc - volume_roc.rolling(window=3).mean()
    
    # Volatility normalization using ATR (Average True Range)
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=5).mean()
    
    # Regime-adaptive smoothing based on volatility regime
    volatility_regime = atr / df['close'].rolling(window=10).std()
    smooth_window = np.where(volatility_regime > volatility_regime.rolling(window=20).median(), 3, 5)
    
    # Core factor: momentum reversal amplified by volume acceleration, normalized by volatility
    raw_factor = momentum_reversal * volume_acceleration / (atr + 1e-7)
    
    # Apply regime-adaptive smoothing
    alpha_factor = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= max(smooth_window):
            window = int(smooth_window[i])
            alpha_factor.iloc[i] = raw_factor.iloc[i-window+1:i+1].mean()
        else:
            alpha_factor.iloc[i] = raw_factor.iloc[i]
    
    # Threshold-based stabilization
    factor_std = alpha_factor.rolling(window=20).std()
    alpha_factor = np.where(abs(alpha_factor) > 2 * factor_std, 
                           np.sign(alpha_factor) * 2 * factor_std, 
                           alpha_factor)
    
    return alpha_factor
