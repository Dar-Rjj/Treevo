import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Novel alpha factor: Adaptive Multi-Timeframe Momentum with Volume-Price Convergence and Volatility-Regime Weighting
    
    This factor employs a sophisticated dynamic weighting system that adapts momentum signals across multiple
    timeframes based on volume-price convergence strength, intraday efficiency patterns, and volatility regime
    characteristics. The factor emphasizes signals with strong confirmation across multiple dimensions while
    downweighting conflicting or weak signals.
    
    Interpretation:
    - High positive values: Strong bullish momentum with robust volume confirmation, efficient intraday action,
      and favorable volatility conditions across multiple timeframes
    - High negative values: Strong bearish momentum with robust volume confirmation, efficient intraday action,
      and favorable volatility conditions across multiple timeframes
    - Values near zero: Mixed signals, weak confirmation, or unfavorable volatility regimes
    
    Key innovations:
    - Four-timeframe momentum capture (1d, 3d, 5d, 8d) for comprehensive trend analysis
    - Dynamic volume-price convergence scoring across multiple confirmation horizons
    - Intraday efficiency patterns that capture momentum-consistent price action
    - Volatility-regime adaptive weighting that emphasizes signals in favorable conditions
    - Multi-dimensional confirmation strength assessment for robust signal selection
    """
    
    # Multi-timeframe momentum components
    momentum_1d = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    momentum_3d = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    momentum_5d = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    momentum_8d = (df['close'] - df['close'].shift(8)) / df['close'].shift(8)
    
    # Volume-price convergence scoring (multiple timeframes and methods)
    # 1-day volume-direction alignment
    price_dir_1d = np.sign(df['close'] - df['close'].shift(1))
    volume_dir_1d = np.sign(df['volume'] - df['volume'].shift(1))
    vol_price_conv_1d = (price_dir_1d == volume_dir_1d).astype(float)
    
    # 3-day volume acceleration with price direction
    vol_ma_3d = df['volume'].rolling(window=3, min_periods=2).mean()
    vol_ma_8d = df['volume'].rolling(window=8, min_periods=5).mean()
    vol_accel_3d = vol_ma_3d / (vol_ma_8d + 1e-7)
    price_dir_3d = np.sign(df['close'] - df['close'].shift(3))
    vol_price_conv_3d = (price_dir_3d == np.sign(vol_accel_3d - 1)).astype(float) * vol_accel_3d
    
    # 5-day volume trend consistency
    vol_trend_5d = df['volume'].rolling(window=5, min_periods=3).apply(
        lambda x: 1 if (x[-1] > x[0] and x[-1] > np.mean(x)) else (-1 if x[-1] < x[0] and x[-1] < np.mean(x) else 0)
    )
    price_trend_5d = np.sign(df['close'] - df['close'].shift(5))
    vol_price_conv_5d = (price_trend_5d == vol_trend_5d).astype(float)
    
    # Combined volume-price convergence score
    vol_price_convergence = (vol_price_conv_1d * 0.4 + vol_price_conv_3d * 0.35 + vol_price_conv_5d * 0.25)
    
    # Intraday efficiency patterns
    intraday_position = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-7)
    
    # Momentum-aligned intraday efficiency
    bullish_efficiency_1d = ((intraday_position > 0.6) & (momentum_1d > 0)).astype(float)
    bearish_efficiency_1d = ((intraday_position < 0.4) & (momentum_1d < 0)).astype(float)
    efficiency_1d = bullish_efficiency_1d - bearish_efficiency_1d
    
    bullish_efficiency_3d = ((intraday_position > 0.65) & (momentum_3d > 0)).astype(float)
    bearish_efficiency_3d = ((intraday_position < 0.35) & (momentum_3d < 0)).astype(float)
    efficiency_3d = bullish_efficiency_3d - bearish_efficiency_3d
    
    combined_efficiency = (efficiency_1d * 0.6 + efficiency_3d * 0.4)
    
    # Range expansion characteristics
    current_range = df['high'] - df['low']
    range_ma_5d = current_range.rolling(window=5, min_periods=3).mean()
    range_expansion = current_range / (range_ma_5d + 1e-7)
    
    # True range volatility calculation
    true_range = pd.DataFrame({
        'hl': df['high'] - df['low'],
        'hcp': abs(df['high'] - df['close'].shift(1)),
        'lcp': abs(df['low'] - df['close'].shift(1))
    }).max(axis=1)
    
    # Multi-timeframe volatility regimes
    vol_3d = true_range.rolling(window=3, min_periods=2).mean()
    vol_8d = true_range.rolling(window=8, min_periods=5).mean()
    vol_regime = vol_3d / (vol_8d + 1e-7)
    
    # Volatility-regime adaptive scoring
    # Favorable conditions: moderate volatility expansion (1.0-1.5x) or low volatility (below 0.8x)
    vol_regime_score = np.where(
        (vol_regime >= 1.0) & (vol_regime <= 1.5), 
        1.2,  # Moderate expansion favorable
        np.where(vol_regime < 0.8, 1.1, 0.8)  # Low vol favorable, high vol unfavorable
    )
    
    # Dynamic confirmation strength assessment
    convergence_strength = np.clip(vol_price_convergence * 1.5, 0, 1)
    efficiency_strength = np.clip(abs(combined_efficiency) * 1.3, 0, 1)
    range_strength = np.clip(range_expansion - 0.8, 0, 1)  # Positive when above average
    
    # Multi-dimensional confirmation score
    confirmation_score = (convergence_strength * 0.4 + efficiency_strength * 0.35 + range_strength * 0.25)
    
    # Timeframe-specific momentum weighting based on confirmation patterns
    # Higher weights for timeframes with stronger confirmation
    base_timeframe_weights = [0.3, 0.25, 0.25, 0.2]  # 1d, 3d, 5d, 8d
    
    # Dynamic adjustment based on confirmation strength and volatility regime
    timeframe_adjustments = [
        confirmation_score * vol_regime_score * 0.7,  # 1d most responsive
        confirmation_score * vol_regime_score * 0.6,  # 3d responsive
        confirmation_score * vol_regime_score * 0.4,  # 5d moderate
        confirmation_score * vol_regime_score * 0.3   # 8d least responsive
    ]
    
    adjusted_weights = [base + adj for base, adj in zip(base_timeframe_weights, timeframe_adjustments)]
    weight_sum = sum(adjusted_weights)
    normalized_weights = [w / weight_sum for w in adjusted_weights]
    
    # Weighted momentum blend with dynamic adjustment
    momentum_blend = (
        momentum_1d * normalized_weights[0] +
        momentum_3d * normalized_weights[1] +
        momentum_5d * normalized_weights[2] +
        momentum_8d * normalized_weights[3]
    )
    
    # Signal amplification based on multi-dimensional confirmation
    amplification_factor = 1 + (confirmation_score * vol_regime_score * 0.8)
    amplified_momentum = momentum_blend * amplification_factor
    
    # Final factor with volatility normalization for cross-sectional consistency
    alpha_factor = amplified_momentum / (vol_8d + 1e-7)
    
    return alpha_factor
