import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Multi-timeframe momentum acceleration with volume divergence using percentile-based regime weights.
    
    Interpretation:
    - Triple-timeframe momentum hierarchy (intraday, overnight, multi-day) with acceleration signals
    - Volume divergence detection between current trading intensity and recent patterns
    - Percentile-based regime classification for adaptive signal weighting
    - Multiplicative combinations enhance signal robustness across market conditions
    - Smooth regime transitions prevent abrupt signal changes
    - Positive values indicate accelerating bullish momentum with volume confirmation
    - Negative values suggest deteriorating momentum with volume distribution patterns
    """
    
    # Core momentum components (no normalization)
    intraday_return = (df['close'] - df['open']) / df['open']
    overnight_return = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    multi_day_return = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    
    # Momentum acceleration signals
    intraday_accel = intraday_return - intraday_return.shift(1)
    overnight_accel = overnight_return - overnight_return.shift(1)
    multi_day_accel = multi_day_return - multi_day_return.shift(2)
    
    # Volume divergence detection
    volume_5d_median = df['volume'].rolling(window=5).median()
    volume_divergence = df['volume'] / (volume_5d_median + 1e-7) - 1.0
    
    # Price range efficiency
    range_efficiency = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-7)
    
    # Percentile-based regime classification
    volatility_20d = (df['high'] - df['low']).rolling(window=20).std()
    vol_percentile = volatility_20d.rolling(window=50).apply(lambda x: (x.iloc[-1] > x.quantile(0.7)) * 2 + (x.iloc[-1] > x.quantile(0.3)) * 1, raw=False)
    
    volume_percentile = df['volume'].rolling(window=50).apply(lambda x: (x.iloc[-1] > x.quantile(0.7)) * 2 + (x.iloc[-1] > x.quantile(0.3)) * 1, raw=False)
    
    # Regime combination matrix
    regime_matrix = vol_percentile * 3 + volume_percentile
    
    # Smooth regime weights using multiplicative scaling
    high_vol_high_volume_weight = 0.8  # Regime 8
    high_vol_medium_volume_weight = 1.2  # Regime 7  
    high_vol_low_volume_weight = 0.6   # Regime 6
    medium_vol_high_volume_weight = 1.4  # Regime 5
    medium_vol_medium_volume_weight = 1.0  # Regime 4
    medium_vol_low_volume_weight = 0.7   # Regime 3
    low_vol_high_volume_weight = 1.1   # Regime 2
    low_vol_medium_volume_weight = 0.9  # Regime 1
    low_vol_low_volume_weight = 0.5    # Regime 0
    
    regime_weights = np.select([
        regime_matrix == 8, regime_matrix == 7, regime_matrix == 6,
        regime_matrix == 5, regime_matrix == 4, regime_matrix == 3,
        regime_matrix == 2, regime_matrix == 1, regime_matrix == 0
    ], [
        high_vol_high_volume_weight, high_vol_medium_volume_weight, high_vol_low_volume_weight,
        medium_vol_high_volume_weight, medium_vol_medium_volume_weight, medium_vol_low_volume_weight,
        low_vol_high_volume_weight, low_vol_medium_volume_weight, low_vol_low_volume_weight
    ], default=1.0)
    
    # Multi-timeframe momentum convergence
    momentum_convergence = (
        intraday_accel * np.sign(intraday_return) +
        overnight_accel * np.sign(overnight_return) +
        multi_day_accel * np.sign(multi_day_return)
    )
    
    # Volume-confirmed acceleration
    volume_confirmed_accel = volume_divergence * momentum_convergence
    
    # Range efficiency multiplier
    efficiency_multiplier = range_efficiency * np.sign(intraday_return)
    
    # Final alpha factor with multiplicative regime adaptation
    alpha_factor = (
        momentum_convergence * 0.4 +
        volume_confirmed_accel * 0.3 +
        efficiency_multiplier * 0.3
    ) * regime_weights
    
    return alpha_factor
