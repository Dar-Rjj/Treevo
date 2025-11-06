import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Advanced alpha factor: Multi-Timeframe Momentum Regime Detection with Adaptive Volatility Scaling
    and Volume-Price Divergence Convergence
    
    Combines short-term (1-day), medium-term (5-day), and long-term (20-day) momentum regimes
    with regime-dependent volatility scaling that adapts to market conditions. Introduces
    volume-price divergence convergence across multiple timeframes to detect when volume
    confirms or contradicts price movements.
    
    Economic rationale: Different momentum timeframes capture distinct market participant behaviors
    (day traders vs. institutional investors). Adaptive volatility scaling provides dynamic
    risk adjustment that responds to changing market regimes. Volume-price divergence convergence
    identifies periods where trading activity strongly supports price trends, enhancing
    the predictive power of momentum signals.
    """
    # Multi-timeframe momentum with regime weighting
    momentum_1d = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    momentum_5d = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    momentum_20d = (df['close'] - df['close'].shift(20)) / df['close'].shift(20)
    
    # Regime detection based on momentum alignment
    momentum_alignment = np.sign(momentum_1d) * np.sign(momentum_5d) * np.sign(momentum_20d)
    regime_weight = 1 + 0.5 * momentum_alignment  # Amplify when all timeframes align
    
    # Adaptive volatility scaling with regime sensitivity
    intraday_vol = (df['high'] - df['low']) / df['close']
    short_term_vol = df['close'].pct_change().rolling(window=5).std()
    medium_term_vol = df['close'].pct_change().rolling(window=10).std()
    
    # Volume-price divergence convergence
    # Short-term: Daily volume vs price movement efficiency
    daily_range = df['high'] - df['low']
    price_efficiency_1d = (df['close'] - df['open']).abs() / (daily_range + 1e-7)
    volume_intensity_1d = df['volume'] / df['volume'].rolling(window=5).mean()
    vp_divergence_1d = price_efficiency_1d * volume_intensity_1d
    
    # Medium-term: 5-day cumulative volume vs price momentum
    cum_volume_5d = df['volume'].rolling(window=5).sum()
    norm_cum_volume_5d = cum_volume_5d / cum_volume_5d.rolling(window=20).mean()
    price_momentum_5d = (df['close'] - df['close'].shift(5)).abs() / df['close'].shift(5)
    vp_divergence_5d = price_momentum_5d * norm_cum_volume_5d
    
    # Long-term: 20-day volume trend vs price trend consistency
    volume_trend_20d = df['volume'].rolling(window=20).apply(
        lambda x: (x[-1] - x[0]) / (x[0] + 1e-7) if x[0] != 0 else 0
    )
    price_trend_consistency_20d = df['close'].rolling(window=20).apply(
        lambda x: len([i for i in range(1, len(x)) if (x[i] - x[i-1]) * (x[i-1] - x[i-2]) > 0]) / 19
        if len(x) == 20 else 0
    )
    vp_divergence_20d = volume_trend_20d * price_trend_consistency_20d
    
    # Combine components with regime-aware weighting
    momentum_blend = 0.4 * momentum_1d + 0.35 * momentum_5d + 0.25 * momentum_20d
    volatility_scaling = intraday_vol + short_term_vol + medium_term_vol
    vp_convergence = vp_divergence_1d * vp_divergence_5d * (1 + vp_divergence_20d)
    
    # Final alpha factor with regime detection and multi-timeframe convergence
    alpha_factor = regime_weight * momentum_blend / (volatility_scaling + 1e-7) * (1 + vp_convergence)
    
    return alpha_factor
