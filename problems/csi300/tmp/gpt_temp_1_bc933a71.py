import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Multi-timeframe momentum acceleration with volatility scaling
    # Clear interpretation: regime-adaptive momentum acceleration × volume confirmation
    
    # Multi-timeframe momentum (5-day and 10-day)
    mom_5 = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    mom_10 = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
    
    # Momentum acceleration (rate of change in momentum)
    mom_accel = mom_5 - mom_10
    
    # Volatility regime detection using rolling IQR (robust to outliers)
    returns = df['close'].pct_change()
    vol_iqr = returns.rolling(window=20, min_periods=10).apply(
        lambda x: np.percentile(x, 75) - np.percentile(x, 25), raw=True
    )
    
    # Volatility-scaled momentum acceleration
    vol_scaled_mom = mom_accel / (vol_iqr + 1e-7)
    
    # Volume confirmation with regime-aware scaling
    volume_ma_short = df['volume'].rolling(window=5, min_periods=3).median()  # robust median
    volume_ma_long = df['volume'].rolling(window=20, min_periods=10).median()
    
    # Volume regime: short-term vs long-term volume ratio
    volume_regime = volume_ma_short / volume_ma_long
    
    # Dynamic volume weighting based on momentum direction
    volume_weight = np.where(mom_accel > 0, volume_regime, 1 / (volume_regime + 1e-7))
    
    # Range efficiency with volatility adjustment
    daily_range = df['high'] - df['low']
    range_normalized = daily_range / df['close'].shift(1)
    range_efficiency = np.abs(returns) / (range_normalized + 1e-7)
    
    # Combine components with dynamic interaction
    alpha_factor = vol_scaled_mom * volume_weight * range_efficiency
    
    return alpha_factor.replace([np.inf, -np.inf], np.nan).fillna(0)
