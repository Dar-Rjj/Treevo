import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Enhanced Regime-Aware Momentum Divergence with Volume-Price Dynamics
    Combines sophisticated momentum divergence patterns with volume-price interaction analysis,
    applies multi-regime smoothing based on volatility and trend regimes,
    and incorporates trend-weighted momentum enhancement for predictive power
    """
    # Enhanced momentum divergence components
    close_momentum = df['close'] / df['close'].shift(5) - 1
    high_momentum = df['high'] / df['high'].shift(5) - 1
    low_momentum = df['low'] / df['low'].shift(5) - 1
    
    # Multi-dimensional momentum divergence
    high_low_divergence = high_momentum - low_momentum
    price_range_divergence = (df['high'] - df['low']) / df['close'].shift(3)
    momentum_asymmetry = (high_momentum + low_momentum) / (2 * close_momentum + 1e-7) - 1
    
    # Enhanced volume-price divergence analysis
    volume_momentum = df['volume'] / df['volume'].rolling(window=8, min_periods=4).mean() - 1
    amount_momentum = df['amount'] / df['amount'].rolling(window=8, min_periods=4).mean() - 1
    
    # Volume-price momentum divergence
    volume_price_divergence = volume_momentum - close_momentum
    amount_price_divergence = amount_momentum - close_momentum
    
    # Volume trend strength using linear regression slope
    volume_trend_strength = df['volume'].rolling(window=12, min_periods=6).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 6 else 0
    )
    
    # Multi-regime detection system
    price_volatility = df['close'].rolling(window=15, min_periods=8).std()
    volatility_regime = price_volatility / price_volatility.rolling(window=40, min_periods=20).mean()
    
    # Trend regime detection using price momentum persistence
    trend_strength = close_momentum.rolling(window=10, min_periods=5).apply(
        lambda x: np.mean(np.sign(x)) if len(x) >= 5 else 0
    )
    
    # Define three volatility regimes
    low_vol_regime = (volatility_regime < 0.7).astype(int)
    high_vol_regime = (volatility_regime > 1.3).astype(int)
    medium_vol_regime = 1 - low_vol_regime - high_vol_regime
    
    # Define trend regimes
    strong_trend_regime = (abs(trend_strength) > 0.4).astype(int)
    weak_trend_regime = (abs(trend_strength) < 0.2).astype(int)
    moderate_trend_regime = 1 - strong_trend_regime - weak_trend_regime
    
    # Core momentum divergence with volume confirmation
    core_divergence = high_low_divergence * (1 + np.sign(high_low_divergence) * volume_price_divergence)
    
    # Enhanced divergence with multiple components
    enhanced_divergence = (
        core_divergence + 
        0.3 * momentum_asymmetry + 
        0.2 * price_range_divergence * np.sign(volume_price_divergence)
    )
    
    # Volume trend weighted momentum enhancement
    volume_trend_weighted = enhanced_divergence * (1 + volume_trend_strength / (abs(volume_trend_strength) + 1e-7))
    
    # Multi-regime smoothing system
    # Volatility-based smoothing windows
    smooth_low_vol = volume_trend_weighted.rolling(window=8, min_periods=4).mean()
    smooth_high_vol = volume_trend_weighted.rolling(window=3, min_periods=2).mean()
    smooth_medium_vol = volume_trend_weighted.rolling(window=5, min_periods=3).mean()
    
    # Trend-based smoothing windows
    smooth_strong_trend = volume_trend_weighted.rolling(window=4, min_periods=2).mean()
    smooth_weak_trend = volume_trend_weighted.rolling(window=10, min_periods=5).mean()
    smooth_moderate_trend = volume_trend_weighted.rolling(window=6, min_periods=3).mean()
    
    # Combine volatility and trend regimes for final smoothing
    volatility_smoothed = (
        low_vol_regime * smooth_low_vol +
        high_vol_regime * smooth_high_vol +
        medium_vol_regime * smooth_medium_vol
    )
    
    trend_smoothed = (
        strong_trend_regime * smooth_strong_trend +
        weak_trend_regime * smooth_weak_trend +
        moderate_trend_regime * smooth_moderate_trend
    )
    
    # Final regime-aware factor combining both smoothing approaches
    regime_aware_factor = 0.6 * volatility_smoothed + 0.4 * trend_smoothed
    
    # Final enhancement with amount-price divergence confirmation
    final_factor = regime_aware_factor * (1 + 0.2 * np.sign(regime_aware_factor) * amount_price_divergence)
    
    return final_factor
