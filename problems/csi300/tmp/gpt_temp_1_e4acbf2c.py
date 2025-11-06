import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Multi-Dimensional Divergence with Adaptive Regime Scaling and Momentum Persistence
    Captures divergence across price, volume, and volatility dimensions with regime-aware scaling,
    incorporates momentum persistence signals, and uses dynamic smoothing based on market conditions
    """
    # Price momentum divergence across multiple timeframes
    momentum_1d = df['close'] / df['close'].shift(1) - 1
    momentum_3d = df['close'] / df['close'].shift(3) - 1
    momentum_7d = df['close'] / df['close'].shift(7) - 1
    
    # Multi-timeframe momentum divergence
    short_term_divergence = momentum_1d - momentum_3d
    medium_term_divergence = momentum_3d - momentum_7d
    momentum_divergence = short_term_divergence + medium_term_divergence
    
    # Volume divergence metrics
    volume_momentum = df['volume'] / df['volume'].shift(3) - 1
    amount_momentum = df['amount'] / df['amount'].shift(3) - 1
    volume_divergence = volume_momentum - amount_momentum
    
    # Price-volume divergence strength
    price_volume_divergence = momentum_divergence * volume_divergence
    
    # Volatility divergence measures
    intraday_range = (df['high'] - df['low']) / df['open']
    close_volatility = (df['close'] - df['close'].shift(1)).abs() / df['close'].shift(1)
    
    # Volatility regime divergence
    short_term_vol = intraday_range.rolling(window=3, min_periods=1).mean()
    medium_term_vol = intraday_range.rolling(window=10, min_periods=1).mean()
    volatility_divergence = short_term_vol / medium_term_vol - 1
    
    # Combined divergence factor
    combined_divergence = price_volume_divergence * volatility_divergence
    
    # Regime detection with multi-dimensional scaling
    volatility_regime = intraday_range / intraday_range.rolling(window=15, min_periods=1).mean()
    volume_regime = df['volume'] / df['volume'].rolling(window=15, min_periods=1).mean()
    momentum_regime = momentum_3d.rolling(window=5, min_periods=1).std()
    
    # Multi-dimensional regime classification
    high_vol_expansion = ((volatility_regime > 1.3) & (volume_regime > 1.1)).astype(float)
    low_vol_breakout = ((volatility_regime < 0.8) & (volume_regime > 1.2) & (momentum_regime > momentum_regime.rolling(window=10, min_periods=1).mean())).astype(float)
    consolidation = ((volatility_regime < 0.7) & (volume_regime < 0.9)).astype(float)
    normal_regime = 1 - high_vol_expansion - low_vol_breakout - consolidation
    
    # Dynamic regime scaling based on market conditions
    regime_scaling = (high_vol_expansion * 0.7 + 
                     low_vol_breakout * 1.5 + 
                     consolidation * 0.6 + 
                     normal_regime * 1.0)
    
    # Momentum persistence signals
    momentum_consistency = momentum_1d.rolling(window=3, min_periods=1).std() * np.sign(momentum_1d.rolling(window=3, min_periods=1).mean())
    trend_acceleration = (momentum_3d - momentum_3d.shift(2)) * momentum_consistency
    
    # Enhanced factor with persistence and regime adaptation
    persistence_enhanced = combined_divergence * (1 + trend_acceleration)
    regime_enhanced_factor = persistence_enhanced * regime_scaling
    
    # Adaptive smoothing based on regime characteristics
    smoothing_window = np.where(high_vol_expansion > 0, 2,
                              np.where(low_vol_breakout > 0, 4,
                                      np.where(consolidation > 0, 6, 3)))
    
    # Apply dynamic smoothing
    final_factor = pd.Series(
        [regime_enhanced_factor.rolling(window=int(w), min_periods=1).mean().iloc[i] 
         for i, w in enumerate(smoothing_window)], 
        index=df.index
    )
    
    return final_factor
