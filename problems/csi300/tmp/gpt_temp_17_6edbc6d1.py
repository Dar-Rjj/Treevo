import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Multi-regime adaptive momentum with volume-flow divergence and volatility-scaled acceleration.
    
    Interpretation:
    - Identifies four market regimes combining volatility and volume conditions
    - Volume-flow divergence captures directional volume pressure independent of price movement
    - Volatility-scaled acceleration adjusts momentum sensitivity based on market turbulence
    - Regime-specific component emphasis optimizes signal quality across different market states
    - Momentum-volume convergence detects alignment between price action and trading activity
    - Positive values indicate bullish momentum with strong volume confirmation in favorable regimes
    - Negative values suggest bearish pressure with distribution patterns in adverse conditions
    - Economic rationale: adaptive regime detection with volume-flow analysis provides robust
      predictive signals by accounting for market microstructure and liquidity conditions
    """
    
    # Core momentum components
    intraday_efficiency = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-7)
    gap_momentum = (df['open'] - df['close'].shift(1)) / (df['high'] - df['low'] + 1e-7)
    daily_efficiency = (df['close'] - df['close'].shift(1)) / (df['high'] - df['low'] + 1e-7)
    swing_momentum = (df['close'] - df['close'].shift(3)) / (
        df['high'].rolling(window=4).max() - df['low'].rolling(window=4).min() + 1e-7
    )
    
    # Volatility regime using adaptive thresholds
    true_range = np.maximum(df['high'] - df['low'], 
                           np.maximum(abs(df['high'] - df['close'].shift(1)), 
                                     abs(df['low'] - df['close'].shift(1))))
    vol_ratio = true_range / (true_range.rolling(window=10).mean() + 1e-7)
    vol_regime = np.where(vol_ratio > 1.8, 'high_vol',
                         np.where(vol_ratio > 1.2, 'medium_vol',
                                 np.where(vol_ratio < 0.8, 'low_vol', 'normal_vol')))
    
    # Volume-flow divergence (directional volume pressure)
    volume_flow = (df['volume'] - df['volume'].shift(1)) * np.sign(df['close'] - df['open'])
    volume_flow_ma = volume_flow.rolling(window=5).mean()
    volume_divergence = (volume_flow - volume_flow_ma) / (abs(volume_flow_ma) + 1e-7)
    
    # Volume regime based on flow characteristics
    volume_persistence = volume_flow.rolling(window=3).apply(lambda x: np.sum(x > 0) - np.sum(x < 0))
    volume_regime = np.where(abs(volume_persistence) == 3, 'directional',
                            np.where(abs(volume_persistence) >= 2, 'biased', 'mixed'))
    
    # Combined regime classification
    full_regime = vol_regime + '_' + volume_regime
    
    # Volatility-scaled momentum acceleration
    momentum_acceleration = (intraday_efficiency + daily_efficiency + swing_momentum) / 3
    vol_scaled_accel = momentum_acceleration * (2.0 - vol_ratio)  # Inverse scaling with volatility
    
    # Volume-flow aligned efficiency
    flow_aligned_efficiency = intraday_efficiency * np.sign(volume_divergence)
    volume_confirmed_momentum = daily_efficiency * (1 + abs(volume_divergence)) * np.sign(daily_efficiency)
    
    # Regime-adaptive component weights
    # High volatility regimes: emphasize volatility-scaled signals and flow alignment
    intraday_weight = np.where((vol_regime == 'high_vol') | (vol_regime == 'medium_vol'), 0.25, 0.35)
    daily_weight = np.where(vol_regime == 'high_vol', 0.15,
                           np.where(vol_regime == 'low_vol', 0.45, 0.3))
    accel_weight = np.where(vol_regime == 'high_vol', 0.35,
                           np.where(vol_regime == 'low_vol', 0.15, 0.25))
    flow_weight = np.where(volume_regime == 'directional', 0.25,
                          np.where(volume_regime == 'biased', 0.2, 0.15))
    
    # Regime intensity multipliers
    regime_intensity = np.where(full_regime == 'high_vol_directional', 1.4,
                               np.where(full_regime == 'low_vol_directional', 1.3,
                                       np.where(full_regime == 'normal_vol_directional', 1.2, 1.0)))
    
    # Combined alpha factor with multi-regime adaptation
    alpha_factor = (
        intraday_weight * intraday_efficiency +
        daily_weight * daily_efficiency +
        accel_weight * vol_scaled_accel +
        flow_weight * (flow_aligned_efficiency + volume_confirmed_momentum) / 2
    ) * regime_intensity
    
    return alpha_factor
