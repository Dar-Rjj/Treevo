import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Multi-timeframe momentum acceleration with volume divergence and percentile-based regime weighting.
    
    Interpretation:
    - Momentum acceleration hierarchy across 3 timeframes (intraday, overnight, multi-day) with regime emphasis
    - Volume divergence detection identifies momentum-volume synchronization/divergence patterns
    - Percentile-based regime classification for robust state transitions
    - Smooth regime weighting using exponential transitions for signal stability
    - Volume-momentum synchronization scoring enhances signal reliability
    - Regime persistence mechanisms reduce noise and improve factor interpretability
    - Positive values indicate synchronized bullish momentum across timeframes with volume confirmation
    - Negative values suggest bearish pressure with volume divergence patterns
    """
    
    # Multi-timeframe momentum components
    intraday_momentum = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-7)
    overnight_momentum = (df['open'] - df['close'].shift(1)) / (df['high'] - df['low'] + 1e-7)
    multiday_momentum = (df['close'] - df['close'].shift(3)) / (df['high'].rolling(3).max() - df['low'].rolling(3).min() + 1e-7)
    
    # Momentum acceleration hierarchy
    ultra_accel = intraday_momentum * np.sign(intraday_momentum + overnight_momentum)
    short_accel = overnight_momentum * np.sign(overnight_momentum + multiday_momentum)
    medium_accel = multiday_momentum * np.sign(multiday_momentum + intraday_momentum)
    
    # Volume divergence detection
    volume_ma_5 = df['volume'].rolling(window=5).mean()
    volume_ma_10 = df['volume'].rolling(window=10).mean()
    volume_divergence = (df['volume'] / (volume_ma_5 + 1e-7)) - (df['volume'] / (volume_ma_10 + 1e-7))
    
    # Percentile-based regime classification
    vol_5d = (df['high'] - df['low']).rolling(window=5).std()
    vol_regime_percentile = vol_5d.rolling(window=20).apply(lambda x: (x.iloc[-1] - x.quantile(0.3)) / (x.quantile(0.7) - x.quantile(0.3) + 1e-7))
    
    # Smooth regime transitions using exponential weighting
    high_vol_weight = 1 / (1 + np.exp(-5 * (vol_regime_percentile - 0.7)))
    low_vol_weight = 1 / (1 + np.exp(5 * (vol_regime_percentile - 0.3)))
    medium_vol_weight = 1 - high_vol_weight - low_vol_weight
    
    # Volume-pressure synchronization scoring
    volume_pressure = df['volume'] / (df['volume'].rolling(window=10).mean() + 1e-7)
    volume_sync_intraday = intraday_momentum * volume_pressure * np.sign(intraday_momentum)
    volume_sync_overnight = overnight_momentum * volume_pressure * np.sign(overnight_momentum)
    volume_sync_multiday = multiday_momentum * volume_pressure * np.sign(multiday_momentum)
    
    # Regime-adaptive momentum weights with smooth transitions
    intraday_w = (0.4 * high_vol_weight + 0.3 * medium_vol_weight + 0.2 * low_vol_weight)
    overnight_w = (0.2 * high_vol_weight + 0.3 * medium_vol_weight + 0.4 * low_vol_weight)
    multiday_w = (0.2 * high_vol_weight + 0.3 * medium_vol_weight + 0.3 * low_vol_weight)
    accel_w = (0.2 * high_vol_weight + 0.1 * medium_vol_weight + 0.1 * low_vol_weight)
    
    # Volume divergence regime adaptation
    volume_div_weight = np.where(volume_divergence > 0.2, 1.2, 
                                np.where(volume_divergence < -0.2, 0.8, 1.0))
    
    # Combined alpha factor with regime persistence
    momentum_base = (
        intraday_w * intraday_momentum +
        overnight_w * overnight_momentum +
        multiday_w * multiday_momentum
    )
    
    momentum_accel = (
        accel_w * (ultra_accel + short_accel + medium_accel) / 3
    )
    
    volume_sync = (
        volume_sync_intraday * intraday_w +
        volume_sync_overnight * overnight_w +
        volume_sync_multiday * multiday_w
    ) / 3
    
    alpha_factor = (
        momentum_base * 0.5 +
        momentum_accel * 0.3 +
        volume_sync * 0.2
    ) * volume_div_weight
    
    return alpha_factor
