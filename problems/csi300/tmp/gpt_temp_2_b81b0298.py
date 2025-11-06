import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Multi-timeframe momentum convergence with volatility-regime adaptive directional coherence.
    
    Interpretation:
    - Clear volatility regime classification using rolling ATR percentiles
    - Directional coherence measures consistency across intraday, daily, and swing timeframes
    - Momentum acceleration captures trend strength changes with regime-sensitive scaling
    - Volume-pressure regimes provide independent confirmation of momentum signals
    - Multiplicative regime-weighting amplifies signals during appropriate market conditions
    - Positive values indicate strong bullish momentum with high directional coherence
    - Negative values suggest bearish pressure with deteriorating momentum dynamics
    - Economic rationale: Stocks showing consistent directional movement across timeframes
      during appropriate volatility regimes tend to exhibit persistent return patterns
    """
    
    # Core momentum components across different timeframes
    daily_range = df['high'] - df['low']
    intraday_momentum = (df['close'] - df['open']) / (daily_range + 1e-7)
    daily_momentum = (df['close'] - df['close'].shift(1)) / (daily_range + 1e-7)
    swing_momentum = (df['close'] - df['close'].shift(5)) / (
        df['high'].rolling(window=5).max() - df['low'].rolling(window=5).min() + 1e-7
    )
    
    # Clear volatility regime classification using ATR percentiles
    true_range = np.maximum(df['high'] - df['low'],
                           np.maximum(abs(df['high'] - df['close'].shift(1)),
                                     abs(df['low'] - df['close'].shift(1))))
    atr_5d = true_range.rolling(window=5).mean()
    
    # Define volatility regimes using percentiles
    vol_regime_low_threshold = atr_5d.rolling(window=20).apply(lambda x: np.percentile(x, 33), raw=True)
    vol_regime_high_threshold = atr_5d.rolling(window=20).apply(lambda x: np.percentile(x, 67), raw=True)
    
    vol_regime = np.where(atr_5d > vol_regime_high_threshold, 'high',
                         np.where(atr_5d > vol_regime_low_threshold, 'medium', 'low'))
    
    # Directional coherence - measure consistency across timeframes
    momentum_signs = pd.DataFrame({
        'intraday': np.sign(intraday_momentum),
        'daily': np.sign(daily_momentum),
        'swing': np.sign(swing_momentum)
    })
    
    # Calculate directional coherence as binary alignment indicator
    directional_coherence = (momentum_signs.abs().sum(axis=1) == 
                           abs(momentum_signs.sum(axis=1))).astype(float)
    
    # Momentum acceleration components
    intraday_accel = intraday_momentum.diff()
    daily_accel = daily_momentum.diff()
    swing_accel = swing_momentum.diff(3)
    
    # Regime-adaptive acceleration sensitivity
    accel_sensitivity = np.where(vol_regime == 'high', 0.6,
                                np.where(vol_regime == 'medium', 1.0, 1.4))
    
    combined_acceleration = (
        intraday_accel * 0.4 +
        daily_accel * 0.4 +
        swing_accel * 0.2
    ) * accel_sensitivity
    
    # Volume-pressure regime classification
    volume_ratio = df['volume'] / (df['volume'].rolling(window=10).mean() + 1e-7)
    volume_regime_low = volume_ratio.rolling(window=20).apply(lambda x: np.percentile(x, 33), raw=True)
    volume_regime_high = volume_ratio.rolling(window=20).apply(lambda x: np.percentile(x, 67), raw=True)
    
    volume_regime = np.where(volume_ratio > volume_regime_high, 'high',
                            np.where(volume_ratio > volume_regime_low, 'medium', 'low'))
    
    # Volume confirmation multipliers
    volume_confirmation = np.where(volume_regime == 'high', 1.4,
                                  np.where(volume_regime == 'low', 0.6, 1.0))
    
    # Regime-specific momentum weighting (multiplicative approach)
    regime_base_weights = {
        'high': {'intraday': 0.15, 'daily': 0.25, 'swing': 0.10, 'accel': 0.50},
        'medium': {'intraday': 0.25, 'daily': 0.30, 'swing': 0.20, 'accel': 0.25},
        'low': {'intraday': 0.30, 'daily': 0.20, 'swing': 0.40, 'accel': 0.10}
    }
    
    # Apply regime-specific weights multiplicatively
    alpha_factor = pd.Series(0.0, index=df.index)
    
    for regime in ['high', 'medium', 'low']:
        regime_mask = (vol_regime == regime)
        if regime_mask.any():
            weights = regime_base_weights[regime]
            regime_factor = (
                weights['intraday'] * intraday_momentum +
                weights['daily'] * daily_momentum +
                weights['swing'] * swing_momentum +
                weights['accel'] * combined_acceleration
            )
            alpha_factor[regime_mask] = regime_factor[regime_mask]
    
    # Final multiplicative combination
    alpha_factor = alpha_factor * directional_coherence * volume_confirmation
    
    return alpha_factor
