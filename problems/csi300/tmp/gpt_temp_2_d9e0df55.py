import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Adaptive momentum factor with volatility-regime switching and volume-price divergence detection.
    
    Interpretation:
    - Positive values: strong momentum with volume confirmation in favorable volatility regimes
    - Negative values: weak momentum or volume-price divergence in adverse conditions
    - Four volatility regimes (very low, low, high, very high) for finer market adaptation
    - Volume-price divergence detects when volume doesn't support price movement
    - Volatility-scaled momentum ensures comparability across different market conditions
    - Price persistence measures sustainability of price movements
    - Dynamic regime weights adapt factor sensitivity to market environment
    """
    
    # 3-day momentum for balanced responsiveness
    momentum_3d = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    
    # Volume-price divergence detection
    volume_ma_3 = df['volume'].rolling(window=3).mean()
    volume_ma_10 = df['volume'].rolling(window=10).mean()
    volume_trend = volume_ma_3 / (volume_ma_10 + 1e-7)
    
    # Detect volume-price divergence (volume trend opposite to price trend)
    volume_divergence = np.where(
        np.sign(momentum_3d) != np.sign(volume_trend - 1),
        -np.abs(momentum_3d),  # Penalty for divergence
        np.abs(momentum_3d)    # Reward for convergence
    )
    
    # Price persistence: consistency of daily moves
    daily_moves = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    move_direction_consistency = daily_moves.rolling(window=3).apply(
        lambda x: np.sum(np.sign(x) == np.sign(np.sum(x))) / len(x) if len(x) == 3 else np.nan
    )
    
    # Volatility regime detection with finer granularity
    daily_returns = df['close'].pct_change()
    volatility_5d = daily_returns.rolling(window=5).std()
    volatility_20d_median = volatility_5d.rolling(window=20).median()
    
    # Four volatility regimes for better adaptation
    very_low_vol = volatility_5d <= volatility_20d_median * 0.5
    low_vol = (volatility_5d > volatility_20d_median * 0.5) & (volatility_5d <= volatility_20d_median * 0.9)
    high_vol = (volatility_5d > volatility_20d_median * 1.1) & (volatility_5d <= volatility_20d_median * 1.5)
    very_high_vol = volatility_5d > volatility_20d_median * 1.5
    
    # Volatility-scaled momentum with regime-aware scaling
    volatility_scaling = np.select(
        [very_low_vol, low_vol, high_vol, very_high_vol],
        [1.2, 1.0, 0.8, 0.6]  # More aggressive in low vol, conservative in high vol
    )
    scaled_momentum = momentum_3d * volatility_scaling
    
    # Regime-specific component weights with dynamic adaptation
    momentum_weight = np.select(
        [very_low_vol, low_vol, high_vol, very_high_vol],
        [0.6, 0.5, 0.4, 0.3]  # Higher momentum weight in calmer markets
    )
    
    divergence_weight = np.select(
        [very_low_vol, low_vol, high_vol, very_high_vol],
        [0.2, 0.25, 0.3, 0.35]  # Higher divergence sensitivity in volatile markets
    )
    
    persistence_weight = np.select(
        [very_low_vol, low_vol, high_vol, very_high_vol],
        [0.2, 0.25, 0.3, 0.35]  # Higher persistence focus in uncertain markets
    )
    
    # Combine components with regime-aware dynamic weights
    alpha_factor = (
        momentum_weight * scaled_momentum +
        divergence_weight * volume_divergence +
        persistence_weight * move_direction_consistency
    )
    
    return alpha_factor
