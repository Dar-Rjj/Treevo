import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Multi-resolution momentum-volume-volatility convergence with adaptive regime scaling.
    
    Economic intuition:
    - Combines momentum persistence across multiple timeframes for robust trend detection
    - Volume-pressure divergence identifies institutional accumulation patterns
    - Volatility regime classification enables adaptive signal amplification/dampening
    - Price efficiency captures intraday momentum quality and directional strength
    
    Interpretation:
    - Positive values: strong bullish momentum with volume confirmation in favorable volatility regimes
    - Negative values: strong bearish momentum with volume divergence in favorable volatility regimes
    - Magnitude reflects combined strength across momentum, volume, volatility, and efficiency dimensions
    """
    
    # Multi-timeframe momentum alignment with geometric weighting
    momentum_2d = df['close'] / df['close'].shift(2) - 1
    momentum_5d = df['close'] / df['close'].shift(5) - 1
    momentum_13d = df['close'] / df['close'].shift(13) - 1
    momentum_21d = df['close'] / df['close'].shift(21) - 1
    
    # Momentum convergence: directional alignment with geometric magnitude scaling
    momentum_direction = np.sign(momentum_2d) * np.sign(momentum_5d) * np.sign(momentum_13d) * np.sign(momentum_21d)
    momentum_magnitude = (abs(momentum_2d) * abs(momentum_5d) * abs(momentum_13d) * abs(momentum_21d)) ** 0.25
    momentum_convergence = momentum_direction * momentum_magnitude
    
    # Volume-pressure divergence with multi-timeframe confirmation
    volume_pressure_short = (df['volume'] - df['volume'].rolling(window=3).mean()) / (df['volume'].rolling(window=3).mean() + 1e-7)
    volume_pressure_medium = (df['volume'] - df['volume'].rolling(window=8).mean()) / (df['volume'].rolling(window=8).mean() + 1e-7)
    
    # Combined volume pressure with directional consistency
    volume_direction = np.sign(volume_pressure_short) * np.sign(volume_pressure_medium)
    volume_magnitude = (abs(volume_pressure_short) * abs(volume_pressure_medium)) ** 0.5
    volume_confirmation = volume_direction * volume_magnitude
    
    # Volatility regime detection using normalized true range
    true_range = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift(1)),
        abs(df['low'] - df['close'].shift(1))
    ], axis=1).max(axis=1)
    
    normalized_volatility = true_range / df['close']
    vol_regime_short = normalized_volatility.rolling(window=5).mean()
    vol_regime_medium = normalized_volatility.rolling(window=13).mean()
    vol_regime_long = normalized_volatility.rolling(window=34).mean()
    
    # Volatility regime quality: stable trending conditions favor signal quality
    volatility_stability = vol_regime_medium / (vol_regime_long + 1e-7)
    volatility_quality = 1.0 / (1.0 + volatility_stability)
    
    # Price efficiency: intraday momentum quality and directional strength
    daily_range = df['high'] - df['low']
    price_efficiency = (df['close'] - df['open']) / (daily_range + 1e-7)
    
    # Adaptive regime classification using volatility percentiles
    vol_percentile_short = vol_regime_short.rolling(window=55).apply(lambda x: (x.iloc[-1] > x.quantile(0.7)).astype(float), raw=False)
    vol_percentile_medium = vol_regime_medium.rolling(window=55).apply(lambda x: (x.iloc[-1] > x.quantile(0.7)).astype(float), raw=False)
    
    # High volatility regime: both short and medium term volatility above 70th percentile
    high_vol_regime = (vol_percentile_short > 0.5) & (vol_percentile_medium > 0.5)
    
    # Low volatility regime: both short and medium term volatility below 30th percentile
    low_vol_regime = (vol_regime_short < vol_regime_short.rolling(window=55).quantile(0.3)) & \
                     (vol_regime_medium < vol_regime_medium.rolling(window=55).quantile(0.3))
    
    # Base factor: multiplicative combination of all components
    base_factor = momentum_convergence * volume_confirmation * price_efficiency * volatility_quality
    
    # Adaptive regime scaling with smooth transitions
    regime_adjusted_factor = base_factor.copy()
    regime_adjusted_factor[low_vol_regime] = base_factor[low_vol_regime] * 1.8  # Amplify in low volatility trending regimes
    regime_adjusted_factor[high_vol_regime] = base_factor[high_vol_regime] * 0.4  # Dampen in high volatility choppy regimes
    
    return regime_adjusted_factor
