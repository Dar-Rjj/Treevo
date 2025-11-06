import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Align timeframes: 5-day for momentum, 10-day for volume, 20-day for volatility
    # Use robust rolling statistics (median, IQR) and regime-aware dynamic weighting
    
    # Price momentum - 5-day return using median-based normalization
    momentum_5d = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    momentum_robust = momentum_5d / (df['close'].pct_change().rolling(window=20, min_periods=10).std() + 1e-7)
    
    # Volume regime detection - compare current volume to multiple timeframes
    volume_short = df['volume'].rolling(window=5, min_periods=3).median()
    volume_long = df['volume'].rolling(window=20, min_periods=10).median()
    volume_regime = (df['volume'] - volume_short) / (volume_long - volume_short + 1e-7)
    
    # Volatility-scaled range efficiency with IQR-based robustness
    true_range = np.maximum(df['high'] - df['low'], 
                           np.maximum(np.abs(df['high'] - df['close'].shift(1)), 
                                     np.abs(df['low'] - df['close'].shift(1))))
    range_efficiency = np.abs(df['close'] - df['close'].shift(1)) / (true_range + 1e-7)
    
    # Volatility regime using IQR-based measure
    returns = df['close'].pct_change()
    vol_iqr = returns.rolling(window=20, min_periods=10).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))
    vol_regime = vol_iqr / (vol_iqr.rolling(window=60, min_periods=30).median() + 1e-7)
    
    # Dynamic weighting based on volatility regime
    low_vol_weight = 1.0 / (1.0 + np.exp(5 * (vol_regime - 0.8)))
    high_vol_weight = 1.0 / (1.0 + np.exp(-5 * (vol_regime - 1.2)))
    
    # Combine components with regime-aware weighting
    base_factor = momentum_robust * volume_regime * range_efficiency
    alpha_factor = base_factor * (1 + low_vol_weight - high_vol_weight)
    
    return alpha_factor.replace([np.inf, -np.inf], np.nan).fillna(0)
