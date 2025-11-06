import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Adaptive momentum-regime synchronization with volume acceleration divergence.
    
    Interpretation:
    - Detects momentum-regime synchronization across multiple timeframes (intraday, daily, weekly)
    - Identifies volume acceleration divergence to confirm momentum sustainability
    - Uses regime-adaptive weights based on volatility clustering and momentum persistence
    - Combines momentum acceleration with volume divergence for enhanced predictive power
    - Positive values indicate synchronized momentum acceleration with volume confirmation
    - Negative values suggest momentum-regime desynchronization or volume divergence
    - Economic rationale: momentum that accelerates across multiple timeframes with confirming
      volume dynamics provides stronger predictive signals for future returns
    """
    
    # Multi-timeframe momentum components
    intraday_momentum = (df['close'] - df['open']) / (df['open'] + 1e-7)
    daily_momentum = (df['close'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-7)
    weekly_momentum = (df['close'] - df['close'].shift(5)) / (df['close'].shift(5) + 1e-7)
    
    # Momentum acceleration signals
    intraday_accel = intraday_momentum - intraday_momentum.shift(1)
    daily_accel = daily_momentum - daily_momentum.shift(2)
    weekly_accel = weekly_momentum - weekly_momentum.shift(3)
    
    # Volume acceleration divergence
    volume_3d_avg = df['volume'].rolling(window=3).mean()
    volume_10d_avg = df['volume'].rolling(window=10).mean()
    volume_accel_short = df['volume'] / (volume_3d_avg + 1e-7)
    volume_accel_long = volume_3d_avg / (volume_10d_avg + 1e-7)
    volume_divergence = volume_accel_short - volume_accel_long
    
    # Regime detection using momentum persistence and volatility
    momentum_persistence = daily_momentum.rolling(window=5).apply(lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) > 1 else 0)
    daily_volatility = (df['high'] - df['low']).rolling(window=5).std() / df['close'].rolling(window=5).mean()
    
    # Regime classification
    trending_regime = (momentum_persistence > momentum_persistence.rolling(window=20).quantile(0.7)) & (daily_volatility < daily_volatility.rolling(window=20).quantile(0.6))
    mean_reverting_regime = (momentum_persistence < momentum_persistence.rolling(window=20).quantile(0.3)) & (daily_volatility > daily_volatility.rolling(window=20).quantile(0.4))
    volatile_regime = (daily_volatility > daily_volatility.rolling(window=20).quantile(0.7))
    
    # Momentum-regime synchronization
    momentum_sync_strength = (
        np.sign(intraday_momentum) * np.sign(daily_momentum) +
        np.sign(intraday_momentum) * np.sign(weekly_momentum) +
        np.sign(daily_momentum) * np.sign(weekly_momentum)
    )
    
    # Acceleration synchronization
    accel_sync_strength = (
        np.sign(intraday_accel) * np.sign(daily_accel) +
        np.sign(intraday_accel) * np.sign(weekly_accel) +
        np.sign(daily_accel) * np.sign(weekly_accel)
    )
    
    # Volume acceleration divergence alignment
    volume_momentum_alignment = volume_divergence * np.sign(intraday_momentum + daily_momentum + weekly_momentum)
    volume_accel_alignment = volume_divergence * np.sign(intraday_accel + daily_accel + weekly_accel)
    
    # Regime-adaptive component weights
    intraday_weight = np.where(trending_regime, 0.35,
                              np.where(mean_reverting_regime, 0.25, 0.3))
    daily_weight = np.where(trending_regime, 0.3,
                           np.where(mean_reverting_regime, 0.3, 0.25))
    weekly_weight = np.where(trending_regime, 0.2,
                            np.where(mean_reverting_regime, 0.35, 0.25))
    sync_weight = np.where(trending_regime, 0.15,
                          np.where(mean_reverting_regime, 0.1, 0.2))
    
    # Volume divergence amplification
    volume_amp = np.where(volume_accel_short > 1.3, 1.3,
                         np.where(volume_accel_short > 1.1, 1.1, 1.0))
    
    # Combined alpha factor with regime adaptation
    alpha_factor = (
        intraday_weight * intraday_accel +
        daily_weight * daily_accel +
        weekly_weight * weekly_accel +
        sync_weight * momentum_sync_strength * accel_sync_strength +
        0.12 * volume_momentum_alignment +
        0.08 * volume_accel_alignment
    ) * volume_amp
    
    return alpha_factor
