import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Regime-Aware Momentum Divergence with Volume Trend Confirmation
    Combines price momentum divergence signals with volume trend analysis,
    adapts to market volatility regimes, and applies regime-aware smoothing
    for more stable factor values across different market conditions
    """
    # Price momentum components with divergence
    close_momentum = df['close'] / df['close'].shift(5) - 1
    high_momentum = df['high'] / df['high'].shift(5) - 1
    low_momentum = df['low'] / df['low'].shift(5) - 1
    
    # Momentum divergence signals
    high_low_divergence = high_momentum - low_momentum
    close_high_divergence = close_momentum - high_momentum
    close_low_divergence = close_momentum - low_momentum
    
    # Combined momentum divergence factor
    momentum_divergence = (high_low_divergence + close_high_divergence + close_low_divergence) / 3
    
    # Volume trend analysis
    volume_trend = df['volume'].rolling(window=10, min_periods=5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 5 else 0
    )
    amount_trend = df['amount'].rolling(window=10, min_periods=5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 5 else 0
    )
    
    # Volume trend confirmation
    volume_confirmation = np.sign(momentum_divergence) * (volume_trend + amount_trend)
    
    # Volatility regime detection
    price_volatility = df['close'].rolling(window=20, min_periods=10).std()
    volume_volatility = df['volume'].rolling(window=20, min_periods=10).std()
    
    # Regime-aware smoothing parameters
    high_vol_regime = (price_volatility > price_volatility.rolling(window=50, min_periods=25).quantile(0.7)) | \
                     (volume_volatility > volume_volatility.rolling(window=50, min_periods=25).quantile(0.7))
    
    # Adaptive smoothing windows based on volatility regime
    smooth_window = np.where(high_vol_regime, 3, 5)
    
    # Apply regime-aware smoothing to momentum divergence
    smoothed_momentum_divergence = pd.Series(
        [momentum_divergence.rolling(window=w, min_periods=1).mean().iloc[i] 
         for i, w in enumerate(smooth_window)], 
        index=momentum_divergence.index
    )
    
    # Volume-adjusted momentum factor
    volume_adjusted_momentum = smoothed_momentum_divergence * (1 + volume_confirmation)
    
    # Volatility normalization (using rolling windows for stability)
    factor_volatility = volume_adjusted_momentum.rolling(window=20, min_periods=10).std()
    
    # Final factor with volatility stabilization
    factor = volume_adjusted_momentum / (factor_volatility + 1e-7)
    
    return factor
