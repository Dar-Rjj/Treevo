import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Multi-timeframe momentum acceleration with volume divergence via percentile-based regime weights.
    
    Interpretation:
    - Triple-timeframe momentum hierarchy (intraday, overnight, multi-day) with acceleration signals
    - Volume divergence detection across different momentum regimes
    - Percentile-based regime classification for smooth transitions between market states
    - Multiplicative combinations enhance signal robustness across different market conditions
    - Volume-momentum synchronization with divergence penalties for contradictory signals
    - Positive values indicate accelerating bullish momentum with volume confirmation
    - Negative values suggest bearish momentum acceleration with volume distribution
    """
    
    # Multi-timeframe momentum components
    intraday_return = (df['close'] - df['open']) / df['open']
    overnight_return = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    daily_return = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Momentum acceleration signals
    intraday_accel = intraday_return - intraday_return.shift(1)
    overnight_accel = overnight_return - overnight_return.shift(1)
    daily_accel = daily_return - daily_return.shift(1)
    
    # Volume divergence components
    volume_5d_ma = df['volume'].rolling(window=5).mean()
    volume_divergence = df['volume'] / (volume_5d_ma + 1e-7)
    
    # Percentile-based regime classification
    vol_20d_percentile = df['volume'].rolling(window=20).apply(
        lambda x: (x.iloc[-1] - x.quantile(0.2)) / (x.quantile(0.8) - x.quantile(0.2) + 1e-7)
    )
    
    price_range_20d_percentile = (df['high'] - df['low']).rolling(window=20).apply(
        lambda x: (x.iloc[-1] - x.quantile(0.2)) / (x.quantile(0.8) - x.quantile(0.2) + 1e-7)
    )
    
    # Smooth regime transitions using sigmoid-like functions
    high_vol_regime = 1 / (1 + np.exp(-10 * (vol_20d_percentile - 0.6)))
    low_vol_regime = 1 / (1 + np.exp(10 * (vol_20d_percentile - 0.4)))
    medium_vol_regime = 1 - high_vol_regime - low_vol_regime
    
    high_range_regime = 1 / (1 + np.exp(-10 * (price_range_20d_percentile - 0.6)))
    low_range_regime = 1 / (1 + np.exp(10 * (price_range_20d_percentile - 0.4)))
    medium_range_regime = 1 - high_range_regime - low_range_regime
    
    # Regime-specific momentum weights
    intraday_weight = (high_vol_regime * 0.2 + medium_vol_regime * 0.4 + low_vol_regime * 0.6)
    overnight_weight = (high_vol_regime * 0.3 + medium_vol_regime * 0.4 + low_vol_regime * 0.5)
    daily_weight = (high_vol_regime * 0.5 + medium_vol_regime * 0.2 + low_vol_regime * 0.1)
    
    # Volume-momentum synchronization with divergence penalties
    intraday_volume_sync = intraday_return * volume_divergence * np.sign(intraday_return)
    overnight_volume_sync = overnight_return * volume_divergence * np.sign(overnight_return)
    daily_volume_sync = daily_return * volume_divergence * np.sign(daily_return)
    
    # Volume divergence penalties for contradictory signals
    volume_penalty = np.where(
        (intraday_return * volume_divergence < 0) | 
        (overnight_return * volume_divergence < 0) | 
        (daily_return * volume_divergence < 0),
        -0.3, 1.0
    )
    
    # Multiplicative momentum acceleration combinations
    momentum_accel_combo = (
        intraday_accel * overnight_accel * daily_accel * 
        np.sign(intraday_accel * overnight_accel * daily_accel)
    )
    
    # Regime-adaptive alpha factor with smooth transitions
    alpha_factor = (
        intraday_weight * intraday_accel +
        overnight_weight * overnight_accel +
        daily_weight * daily_accel +
        high_range_regime * momentum_accel_combo * 0.4 +
        medium_range_regime * momentum_accel_combo * 0.2 +
        low_range_regime * momentum_accel_combo * 0.1 +
        (intraday_volume_sync + overnight_volume_sync + daily_volume_sync) * 0.15
    ) * volume_penalty
    
    return alpha_factor
