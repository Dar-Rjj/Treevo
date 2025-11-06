import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Hierarchical momentum acceleration with volume divergence and percentile-based regime adaptation.
    
    Interpretation:
    - Multi-timeframe momentum hierarchy (intraday, overnight, weekly) with acceleration signals
    - Volume divergence detection across different time horizons for pressure confirmation
    - Dynamic regime classification using percentile ranks for robust regime identification
    - Multiplicative combination of momentum and volume components for enhanced signal strength
    - Hierarchical weighting system that adapts to market conditions through percentile thresholds
    - Positive values indicate strong bullish momentum with volume confirmation across multiple timeframes
    - Negative values suggest bearish pressure with volume distribution divergence patterns
    """
    
    # Hierarchical momentum components with acceleration
    intraday_momentum = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-7)
    overnight_momentum = (df['open'] - df['close'].shift(1)) / (df['high'] - df['low'] + 1e-7)
    weekly_momentum = (df['close'] - df['close'].shift(5)) / (df['high'].rolling(5).max() - df['low'].rolling(5).min() + 1e-7)
    
    # Momentum acceleration hierarchy
    intraday_accel = intraday_momentum - intraday_momentum.shift(1)
    overnight_accel = overnight_momentum - overnight_momentum.shift(1)
    weekly_accel = weekly_momentum - weekly_momentum.shift(1)
    
    # Volume divergence components across timeframes
    volume_short = df['volume'] / (df['volume'].rolling(window=3).mean() + 1e-7)
    volume_medium = df['volume'] / (df['volume'].rolling(window=8).mean() + 1e-7)
    volume_long = df['volume'] / (df['volume'].rolling(window=21).mean() + 1e-7)
    
    # Volume divergence signals
    volume_divergence_short = volume_short * np.sign(intraday_momentum)
    volume_divergence_medium = volume_medium * np.sign(overnight_momentum)
    volume_divergence_long = volume_long * np.sign(weekly_momentum)
    
    # Percentile-based regime classification
    vol_5d = (df['high'] - df['low']).rolling(window=5).std()
    vol_percentile = vol_5d.rolling(window=20).apply(lambda x: (x.iloc[-1] > np.percentile(x, 70)) * 2 + 
                                                    (x.iloc[-1] > np.percentile(x, 30)) * 1, raw=False)
    
    volume_percentile = df['volume'].rolling(window=20).apply(lambda x: (x.iloc[-1] > np.percentile(x, 70)) * 2 + 
                                                             (x.iloc[-1] > np.percentile(x, 30)) * 1, raw=False)
    
    # Dynamic regime weights using percentile combinations
    regime_weights = np.where((vol_percentile == 3) & (volume_percentile == 3), 2.0,
                             np.where((vol_percentile == 3) & (volume_percentile == 2), 1.8,
                                     np.where((vol_percentile == 2) & (volume_percentile == 3), 1.6,
                                             np.where((vol_percentile == 2) & (volume_percentile == 2), 1.4,
                                                     np.where((vol_percentile == 1) & (volume_percentile >= 2), 1.2, 1.0)))))
    
    # Multiplicative combination of momentum and volume components
    momentum_volume_product = (
        intraday_momentum * volume_divergence_short *
        overnight_momentum * volume_divergence_medium *
        weekly_momentum * volume_divergence_long
    ) ** (1/6)  # Geometric mean for multiplicative combination
    
    # Acceleration hierarchy with regime adaptation
    accel_combined = (
        intraday_accel * regime_weights * 0.4 +
        overnight_accel * regime_weights * 0.3 +
        weekly_accel * regime_weights * 0.3
    )
    
    # Hierarchical alpha factor construction
    alpha_factor = (
        momentum_volume_product * 0.6 +
        accel_combined * 0.4
    ) * regime_weights
    
    return alpha_factor
