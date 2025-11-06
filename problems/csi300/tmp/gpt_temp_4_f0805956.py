import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Multi-timeframe momentum acceleration with volume divergence and percentile-based regime weighting.
    
    Interpretation:
    - Momentum acceleration hierarchy across 3 timeframes (intraday, overnight, multi-day) with regime-specific emphasis
    - Volume divergence detection identifies momentum-validated vs momentum-contradicted periods
    - Percentile-based regime classification for robust state transitions
    - Smooth regime weighting using percentile distances for stable signal transitions
    - Volume-momentum synchronization enhances signal reliability during regime persistence
    - Positive values indicate accelerating bullish momentum with volume confirmation
    - Negative values suggest bearish momentum acceleration with volume distribution
    """
    
    # Multi-timeframe momentum components
    intraday_momentum = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-7)
    overnight_momentum = (df['open'] - df['close'].shift(1)) / (df['high'] - df['low'] + 1e-7)
    multiday_momentum = (df['close'] - df['close'].shift(3)) / (df['high'].rolling(3).max() - df['low'].rolling(3).min() + 1e-7)
    
    # Momentum acceleration hierarchy
    ultra_short_accel = (intraday_momentum + overnight_momentum) * np.sign(intraday_momentum * overnight_momentum)
    medium_term_accel = multiday_momentum * np.sign(ultra_short_accel + intraday_momentum)
    combined_accel = ultra_short_accel + medium_term_accel * np.sign(ultra_short_accel * medium_term_accel)
    
    # Volume divergence detection
    volume_ma_5 = df['volume'].rolling(window=5).mean()
    volume_ma_10 = df['volume'].rolling(window=10).mean()
    volume_divergence = (df['volume'] - volume_ma_5) / (volume_ma_10 + 1e-7)
    
    # Momentum-volume synchronization
    momentum_volume_sync = intraday_momentum * volume_divergence * np.sign(intraday_momentum * volume_divergence)
    acceleration_volume_sync = combined_accel * volume_divergence * np.sign(combined_accel * volume_divergence)
    
    # Percentile-based regime classification
    momentum_strength = intraday_momentum.abs() + overnight_momentum.abs() + multiday_momentum.abs()
    momentum_percentile = momentum_strength.rolling(window=20).apply(lambda x: (x.iloc[-1] - x.quantile(0.3)) / (x.quantile(0.7) - x.quantile(0.3) + 1e-7))
    
    volume_strength = volume_divergence.abs()
    volume_percentile = volume_strength.rolling(window=20).apply(lambda x: (x.iloc[-1] - x.quantile(0.3)) / (x.quantile(0.7) - x.quantile(0.3) + 1e-7))
    
    # Smooth regime weighting using percentile distances
    momentum_regime_weight = np.where(momentum_percentile > 0.6, 1.4,
                                     np.where(momentum_percentile < 0.4, 0.6, 1.0))
    
    volume_regime_weight = np.where(volume_percentile > 0.6, 1.3,
                                   np.where(volume_percentile < 0.4, 0.7, 1.0))
    
    # Regime persistence detection
    momentum_regime_persistence = (momentum_percentile > 0.6).rolling(window=3).sum() / 3
    volume_regime_persistence = (volume_percentile > 0.6).rolling(window=3).sum() / 3
    
    # Enhanced regime weights with persistence bonus
    momentum_weight_enhanced = momentum_regime_weight * (1 + 0.2 * momentum_regime_persistence)
    volume_weight_enhanced = volume_regime_weight * (1 + 0.15 * volume_regime_persistence)
    
    # Combined alpha factor with percentile-based regime adaptation
    alpha_factor = (
        momentum_weight_enhanced * (
            0.35 * intraday_momentum +
            0.25 * overnight_momentum +
            0.20 * multiday_momentum +
            0.20 * combined_accel
        ) +
        volume_weight_enhanced * (
            0.15 * momentum_volume_sync +
            0.10 * acceleration_volume_sync
        )
    )
    
    return alpha_factor
