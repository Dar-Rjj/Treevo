import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Multi-timeframe momentum acceleration with volume divergence via percentile-based regime weights.
    
    Interpretation:
    - Uses percentile-based regime classification for robust regime detection
    - Combines momentum acceleration across intraday, overnight, and daily timeframes
    - Incorporates volume divergence to confirm momentum signals
    - Employs smooth transitions between regimes using multiplicative combinations
    - Volume-momentum synchronization enhances signal reliability
    - Positive values indicate bullish momentum with volume confirmation
    - Negative values suggest bearish pressure with volume divergence
    """
    
    # Core momentum components (normalized by price range)
    intraday_momentum = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-7)
    overnight_momentum = (df['open'] - df['close'].shift(1)) / (df['high'] - df['low'] + 1e-7)
    daily_momentum = (df['close'] - df['close'].shift(1)) / (df['high'] - df['low'] + 1e-7)
    
    # Momentum acceleration hierarchy
    ultra_short_accel = (intraday_momentum + overnight_momentum) * np.sign(intraday_momentum * overnight_momentum)
    daily_accel = daily_momentum * np.sign(intraday_momentum + overnight_momentum)
    combined_accel = ultra_short_accel + daily_accel * np.sign(ultra_short_accel * daily_accel)
    
    # Volume divergence components
    volume_ratio = df['volume'] / (df['volume'].rolling(window=10).mean() + 1e-7)
    amount_ratio = df['amount'] / (df['amount'].rolling(window=10).mean() + 1e-7)
    volume_divergence = volume_ratio - amount_ratio
    
    # Percentile-based regime classification
    daily_range = df['high'] - df['low']
    vol_5d = daily_range.rolling(window=5).std()
    vol_percentile = vol_5d.rolling(window=20).apply(lambda x: (x.iloc[-1] > x.quantile(0.7)) * 2 + (x.iloc[-1] > x.quantile(0.3)) * 1)
    
    volume_percentile = volume_ratio.rolling(window=20).apply(lambda x: (x.iloc[-1] > x.quantile(0.7)) * 2 + (x.iloc[-1] > x.quantile(0.3)) * 1)
    
    # Regime persistence weights
    vol_regime_persistence = vol_percentile.rolling(window=3).mean()
    volume_regime_persistence = volume_percentile.rolling(window=3).mean()
    
    # Multiplicative regime combinations
    regime_multiplier = (1 + 0.2 * vol_regime_persistence) * (1 + 0.15 * volume_regime_persistence)
    
    # Volume-momentum synchronization
    volume_momentum_sync = volume_divergence * np.sign(intraday_momentum + daily_momentum)
    volume_confirmed_momentum = (intraday_momentum + daily_momentum) * np.abs(volume_momentum_sync)
    
    # Dynamic weights based on regime persistence
    intraday_weight = 0.4 * (1 + 0.1 * vol_regime_persistence)
    overnight_weight = 0.2 * (1 + 0.05 * volume_regime_persistence)
    daily_weight = 0.3 * (1 + 0.08 * (vol_regime_persistence + volume_regime_persistence))
    accel_weight = 0.1 * (1 + 0.12 * vol_regime_persistence)
    
    # Combined alpha factor with smooth transitions
    alpha_factor = (
        intraday_weight * intraday_momentum +
        overnight_weight * overnight_momentum +
        daily_weight * daily_momentum +
        accel_weight * combined_accel +
        0.15 * volume_confirmed_momentum * regime_multiplier
    )
    
    return alpha_factor
