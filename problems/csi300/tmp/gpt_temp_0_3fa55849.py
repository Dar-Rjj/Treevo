import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Multi-timeframe momentum acceleration with volume divergence using percentile-based regime weights.
    
    Interpretation:
    - Triple-timeframe momentum hierarchy (intraday, overnight, weekly) with acceleration signals
    - Volume divergence detection using percentile-based regime classification
    - Multiplicative combinations enhance signal robustness across market conditions
    - Smooth regime transitions using percentile thresholds for stable factor behavior
    - Positive values indicate bullish momentum with volume confirmation across timeframes
    - Negative values suggest bearish pressure with volume divergence patterns
    """
    
    # Multi-timeframe momentum components
    intraday_momentum = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-7)
    overnight_momentum = (df['open'] - df['close'].shift(1)) / (df['high'] - df['low'] + 1e-7)
    weekly_momentum = (df['close'] - df['close'].shift(5)) / (df['high'].rolling(5).max() - df['low'].rolling(5).min() + 1e-7)
    
    # Momentum acceleration signals
    intraday_accel = intraday_momentum - intraday_momentum.shift(1)
    overnight_accel = overnight_momentum - overnight_momentum.shift(1)
    weekly_accel = weekly_momentum - weekly_momentum.shift(1)
    
    # Volume divergence detection using percentile regimes
    volume_ratio = df['volume'] / (df['volume'].rolling(window=10).mean() + 1e-7)
    volume_percentile = volume_ratio.rolling(window=20).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    
    # Volume regime classification using smooth percentile thresholds
    volume_regime = np.where(volume_percentile > 0.8, 'high_volume',
                            np.where(volume_percentile < 0.3, 'low_volume', 'normal_volume'))
    
    # Momentum acceleration hierarchy with multiplicative combinations
    short_term_accel = intraday_accel * overnight_accel * np.sign(intraday_accel + overnight_accel)
    medium_term_accel = overnight_accel * weekly_accel * np.sign(overnight_accel + weekly_accel)
    combined_accel = short_term_accel * medium_term_accel * np.sign(short_term_accel + medium_term_accel)
    
    # Volume divergence signals
    volume_momentum_divergence = volume_ratio * (intraday_momentum - weekly_momentum)
    volume_accel_divergence = volume_ratio * (intraday_accel - weekly_accel)
    
    # Percentile-based regime weights for smooth transitions
    intraday_weight = np.where(volume_regime == 'high_volume', 0.4,
                              np.where(volume_regime == 'low_volume', 0.2, 0.3))
    
    overnight_weight = np.where(volume_regime == 'high_volume', 0.3,
                               np.where(volume_regime == 'low_volume', 0.3, 0.25))
    
    weekly_weight = np.where(volume_regime == 'high_volume', 0.2,
                            np.where(volume_regime == 'low_volume', 0.4, 0.3))
    
    accel_weight = np.where(volume_regime == 'high_volume', 0.1,
                           np.where(volume_regime == 'low_volume', 0.1, 0.15))
    
    # Multiplicative alpha factor with volume divergence
    alpha_factor = (
        intraday_weight * intraday_momentum * np.sign(intraday_accel) +
        overnight_weight * overnight_momentum * np.sign(overnight_accel) +
        weekly_weight * weekly_momentum * np.sign(weekly_accel) +
        accel_weight * combined_accel +
        0.05 * volume_momentum_divergence * np.sign(combined_accel) +
        0.03 * volume_accel_divergence * np.sign(combined_accel)
    )
    
    return alpha_factor
